#!/usr/bin/env python3
"""Base preflight, backend configuration, and opt-in managed WhisperX and motion-analysis installs."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import (CONFIG_DIR, CONFIG_FILE, ConfigError, DETAILS, ENGINES, get_config,
                    load_api_key, load_gemini_key, read_env_file, resolve_engine, write_settings)
from runtime import configure_stdio, diagnostic, run_text
import motion

REQUIRED_BINARIES = ['ffmpeg', 'ffprobe', 'yt-dlp']
WHISPERX_VERSION = '3.8.6'
_PERM_WARNED: set[str] = set()
ENV_TEMPLATE = '''# /watch configuration. No shell interpolation; last assignment wins.
# Native captions always come first. Optional fallback: auto|whisperx|groq|openai|none.
# auto preserves the Groq-then-OpenAI preference for existing installations.
# The first-run skill wizard sets WATCH_DETAIL and WATCH_WHISPER_BACKEND.
# WATCH_ENGINE=auto|gemini|local. auto uses Gemini when GEMINI_API_KEY is set.
# With the gemini engine, local videos are uploaded to Google for analysis.
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
'''


def _which(name):
    return shutil.which(name)


def _check_binaries():
    return [name for name in REQUIRED_BINARIES if not _which(name)]


def _check_file_permissions(path: Path) -> None:
    # Native Windows mode bits do not describe NTFS ACLs. WSL Linux homes do.
    if os.name == 'nt' or platform.system() == 'Windows' or str(path) in _PERM_WARNED:
        return
    try:
        if path.stat().st_mode & 0o044:
            _PERM_WARNED.add(str(path))
            print(f'[watch] WARNING: {path} is readable by other users. Run: chmod 600 "{path}"', file=sys.stderr)
    except OSError:
        pass


def _have_api_key():
    backend, key = load_api_key()
    return bool(key), backend


def is_first_run():
    return read_env_file(CONFIG_FILE).get('SETUP_COMPLETE') != 'true'


def _scaffold_env():
    if CONFIG_FILE.exists():
        read_env_file(CONFIG_FILE)  # Fail safely before modifying bad config.
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(ENV_TEMPLATE)
    return True


def _write_setup_complete():
    _scaffold_env()
    write_settings({'SETUP_COMPLETE': 'true'}, CONFIG_FILE)


def _brew_pkg(missing):
    return list(dict.fromkeys('ffmpeg' if name == 'ffprobe' else name for name in missing))


def _install_step(cmd, step):
    print(f"[setup] {step}: {' '.join(map(str, cmd))}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(list(map(str, cmd)))
    except OSError as exc:
        raise SystemExit(f'{step}: cannot run {cmd[0]} ({type(exc).__name__}); install or repair that executable, then retry.') from None
    if result.returncode:
        raise SystemExit(f'{step} failed (exit {result.returncode}): retry setup after checking the error above. '
                         'For WhisperX, allow at least 3 GB free disk and 8 GB RAM; package/model downloads need network access.')


def _install_macos(missing):
    if not _which('brew'):
        return False, 'Homebrew is missing. Install it from https://brew.sh, then rerun setup.'
    _install_step(['brew', 'install', *_brew_pkg(missing)], 'Install media dependencies')
    return True, 'Installed media dependencies with Homebrew.'


def _install_hint_linux(missing):
    hints = []
    if 'ffmpeg' in _brew_pkg(missing):
        hints.append('apt: sudo apt install ffmpeg (or the equivalent package for your distribution)')
    if 'yt-dlp' in missing:
        hints.append('Install pipx, then: pipx install "yt-dlp[default,curl-cffi]"; pipx ensurepath. For YouTube also install Deno: https://deno.com/')
    return '\n'.join(hints)


def _install_hint_windows(missing):
    hints = []
    if 'ffmpeg' in _brew_pkg(missing):
        hints.append('winget install --id Gyan.FFmpeg --exact')
    if 'yt-dlp' in missing:
        hints.extend(['winget install --id yt-dlp.yt-dlp --exact', 'winget install --id DenoLand.Deno --exact'])
    return '\n'.join(hints) + '\nReopen your terminal/agent after PATH changes.'


def _probe(cmd):
    try:
        result = run_text(cmd, timeout=20)
        output = result.stdout or result.stderr
        return {'ok': result.returncode == 0, 'version_line': output.splitlines()[0] if output else None, 'output': diagnostic(output, 2500)}
    except SystemExit as exc:
        return {'ok': False, 'output': str(exc)}


def _whisperx_status(cfg, detailed=False):
    from local_whisperx import sentinel_for
    value = cfg['whisperx_bin']
    executable = Path(value) if value else None
    ready = bool(executable and executable.is_absolute() and executable.is_file() and sentinel_for(executable).is_file())
    if ready and detailed:
        # Explicit detailed diagnostics may start the CLI. --check never does.
        try:
            from local_whisperx import child_env
            result = run_text([str(executable), '--help'], timeout=30, env=child_env())
            ready = result.returncode == 0 and '--no_align' in result.stdout
        except SystemExit:
            ready = False
    return ready


def _status(detailed=False):
    missing = _check_binaries()
    _check_file_permissions(CONFIG_FILE)
    cfg = get_config()
    gemini_key = bool(load_gemini_key())
    try:
        engine = resolve_engine(cfg['engine'], gemini_key)
    except ConfigError:
        engine = 'gemini'  # Explicitly chosen but the key is missing; gemini_key_present reports it.
    binaries_required = engine == 'local'
    has_key, detected = _have_api_key()
    chosen = cfg['whisper_backend']
    backend = detected if chosen == 'auto' else chosen
    local_ready = _whisperx_status(cfg, detailed)
    backend_ready = local_ready if chosen == 'whisperx' else (bool(load_api_key(backend)[1]) if backend in ('groq', 'openai') else chosen == 'none')
    blocked = bool(missing) and binaries_required
    result = {'status': 'needs_install' if blocked else 'ready', 'can_proceed': not blocked,
              'engine': engine, 'configured_engine': cfg['engine'], 'gemini_key_present': gemini_key,
              'gemini_model': cfg['gemini_model'], 'binaries_required': binaries_required,
              'first_run': is_first_run(), 'setup_complete': not is_first_run(),
              'missing_binaries': missing, 'whisper_backend': backend, 'configured_backend': chosen,
              'has_api_key': has_key, 'backend_ready': backend_ready,
              'motion_ready': motion.is_ready(), 'tracker_ready': motion.tracker_ready(), 'track_model_ready': motion.MODEL_PATH.is_file(), 'whisperx_ready': local_ready, 'whisperx_bin': cfg['whisperx_bin'],
              'whisperx_model': cfg['whisperx_model'], 'config_file': str(CONFIG_FILE),
              'watch_detail': cfg['detail'], 'platform': platform.system()}
    if detailed:
        tools = {}
        for name in REQUIRED_BINARIES:
            path = _which(name)
            tools[name] = {'path': path, **(_probe([path, '--version' if name == 'yt-dlp' else '-version']) if path else {'ok': False})}
        ytdlp = _which('yt-dlp')
        result['tools'] = tools
        result['youtube'] = {
            'deno': _which('deno'), 'node': _which('node'),
            'ejs': 'unknown (inspect the owning yt-dlp installation; no network probe)',
            'impersonation': _probe([ytdlp, '--ignore-config', '--list-impersonate-targets']) if ytdlp else None,
            'update_hint': 'Update the owning package: brew upgrade yt-dlp; pipx upgrade yt-dlp; or winget upgrade --id yt-dlp.yt-dlp --exact. Verify the executable path above.',
        }
    return result


def cmd_check():
    # Optional credentials and local-model readiness never block base watch.
    status = _status()
    if status['can_proceed']:
        return 0
    print(f"[watch] Missing {', '.join(status['missing_binaries'])}. Run python3 \"{Path(__file__).resolve()}\".", file=sys.stderr)
    return 2


def cmd_json():
    print(json.dumps(_status(detailed=True), indent=2))
    return 0


def cmd_install(backend=None, detail=None, engine=None):
    if engine:
        _scaffold_env()
        write_settings({'WATCH_ENGINE': engine}, CONFIG_FILE)
    if engine == 'gemini':
        if not load_gemini_key():
            print(f'[setup] Add GEMINI_API_KEY privately to {CONFIG_FILE} (free key: https://aistudio.google.com/apikey), '
                  'then rerun --engine gemini. Local videos will be uploaded to Google for analysis.', file=sys.stderr)
            return 3
        _write_setup_complete()
        missing = _check_binaries()
        note = (f" Optional: install {', '.join(_brew_pkg(missing))} for non-YouTube URLs and --engine local." if missing else '')
        print(f'[setup] Gemini engine is ready. Configuration: {CONFIG_FILE}.{note}')
        return 0
    missing = _check_binaries()
    if missing:
        system = platform.system()
        if system == 'Darwin':
            ok, message = _install_macos(missing)
            print(f'[setup] {message}', file=sys.stderr)
            if not ok or _check_binaries():
                return 2
        else:
            print(_install_hint_windows(missing) if system == 'Windows' else _install_hint_linux(missing), file=sys.stderr)
            return 2
    _scaffold_env()
    if detail:
        write_settings({'WATCH_DETAIL': detail}, CONFIG_FILE)
    if backend == 'whisperx':
        return install_whisperx()
    if backend:
        write_settings({'WATCH_WHISPER_BACKEND': backend}, CONFIG_FILE)
        if backend in ('groq', 'openai') and not load_api_key(backend)[1]:
            print(f'[setup] Base watch is ready. Add {backend.upper()}_API_KEY privately to {CONFIG_FILE}, then rerun --backend {backend}.', file=sys.stderr)
            return 3
        _write_setup_complete()
    print(f'[setup] Base watch is ready. Configuration: {CONFIG_FILE}')
    return 0


def _ensure_uv():
    name = 'uv.exe' if platform.system() == 'Windows' else 'uv'
    installed = _which('uv')
    fallback = Path.home() / '.local' / 'bin' / name
    if installed or fallback.is_file():
        return installed or str(fallback)
    system = platform.system()
    if system == 'Darwin':
        if not _which('brew'):
            raise SystemExit('WhisperX setup needs uv. Install Homebrew from https://brew.sh, then rerun setup.')
        _install_step(['brew', 'install', 'uv'], 'Install uv')
    elif system == 'Windows':
        _install_step(['powershell', '-NoProfile', '-ExecutionPolicy', 'ByPass', '-c', 'irm https://astral.sh/uv/install.ps1 | iex'], 'Install uv')
    elif system == 'Linux':
        # Fetch first: a failed curl must not be hidden by a successful empty sh.
        with tempfile.TemporaryDirectory(prefix='watch-uv-') as temporary:
            script = Path(temporary) / 'install.sh'
            _install_step(['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '-o', script], 'Download uv installer')
            _install_step(['sh', script], 'Install uv')
    else:
        raise SystemExit('Install uv from https://docs.astral.sh/uv/ before retrying WhisperX setup.')
    installed = _which('uv')
    if installed or fallback.is_file():
        return installed or str(fallback)
    raise SystemExit('uv installed but could not be found; reopen your terminal and rerun setup.')


def install_whisperx():
    from local_whisperx import settings, transcribe_audio
    print('[setup] Local WhisperX: requires 3 GB free disk, 8 GB RAM, a 64-bit CPU, and network access for setup. '
          'Downloads about 1 GB of packages plus the 464 MB small model. macOS Apple Silicon is verified; '
          'Intel macOS, Linux, Windows, and CUDA installs are untested.', file=sys.stderr, flush=True)
    if not _which('ffmpeg'):
        raise SystemExit('Install FFmpeg first, then rerun setup.py --install-whisperx.')
    cfg = settings()
    # Check the existing config before downloading or modifying an environment.
    _scaffold_env()
    uv = _ensure_uv()
    venv = Path.home() / '.cache' / 'watch' / 'whisperx-venv'
    sentinel = venv / '.deps-ok'
    windows = platform.system() == 'Windows'
    python = venv / ('Scripts/python.exe' if windows else 'bin/python')
    executable = venv / ('Scripts/whisperx.exe' if windows else 'bin/whisperx')
    ready = sentinel.is_file() and python.is_file() and executable.is_file()
    if not ready:
        if venv.is_symlink():
            raise SystemExit('Managed WhisperX venv is a symlink; choose a regular managed directory before installing.')
        if venv.exists():
            shutil.rmtree(venv)  # Only this fixed, installer-owned environment.
        venv.parent.mkdir(parents=True, exist_ok=True)
        _install_step([uv, 'venv', str(venv), '--python', '3.12'], 'Create WhisperX Python 3.12 environment')
        if platform.system() in ('Linux', 'Windows'):
            _install_step([uv, 'pip', 'install', '--python', python, 'torch==2.8.0', 'torchaudio==2.8.0',
                           '--index-url', 'https://download.pytorch.org/whl/cpu'], 'Install CPU PyTorch wheels')
        _install_step([uv, 'pip', 'install', '--python', python, f'whisperx=={WHISPERX_VERSION}'], 'Install WhisperX')
    # Invalidate before warm-up: an interrupted repair is always safe to rerun.
    sentinel.unlink(missing_ok=True)
    cfg['whisperx_bin'] = str(executable)
    with tempfile.TemporaryDirectory(prefix='watch-warmup-') as temporary:
        audio = Path(temporary) / 'silence.mp3'
        _install_step(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                       'anullsrc=r=16000:cl=mono', '-t', '2', '-acodec', 'libmp3lame', str(audio)], 'Create warm-up audio')
        transcribe_audio(audio, cfg, require_ready=False)
    resolved = run_text([str(uv), 'pip', 'list', '--python', str(python), '--format', 'json'], timeout=60)
    if resolved.returncode:
        raise SystemExit(f'Could not record the WhisperX install: {diagnostic(resolved.stderr)}')
    packages = json.loads(resolved.stdout)
    (venv / 'watch-install.json').write_text(json.dumps({'whisperx': WHISPERX_VERSION, 'model': cfg['whisperx_model'],
                                                       'platform': platform.system(), 'packages': packages}, indent=2), encoding='utf-8')
    sentinel.write_text(WHISPERX_VERSION + '\n', encoding='utf-8')
    write_settings({'WATCH_WHISPER_BACKEND': 'whisperx', 'WATCH_WHISPERX_BIN': str(executable),
                    'WATCH_WHISPERX_MODEL': cfg['whisperx_model'], 'SETUP_COMPLETE': 'true'}, CONFIG_FILE)
    print('[setup] WhisperX is ready; packages and both model caches are warmed.', file=sys.stderr)
    return 0


def _fetch_track_model():
    """The 0.7 MB OpenCV-zoo ViT tracker, pinned by SHA-256; the object tools fall back to template matching without it."""
    import hashlib
    from urllib.request import urlopen
    target = motion.MODEL_PATH
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == motion.MODEL_SHA256:
        return True
    print(f'[setup] Downloading the ViT object-tracking model (0.7 MB) from {motion.MODEL_URL}', file=sys.stderr, flush=True)
    try:
        with urlopen(motion.MODEL_URL, timeout=60) as response:
            data = response.read(8_000_000)
    except OSError as exc:
        print(f'[setup] Model download failed ({type(exc).__name__}); object tracking will use template matching. Rerun --install-motion to retry.', file=sys.stderr)
        return False
    if hashlib.sha256(data).hexdigest() != motion.MODEL_SHA256:
        print('[setup] Downloaded model failed its checksum and was discarded.', file=sys.stderr)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True


def install_motion():
    """Managed venv (numpy, OpenCV, SciPy) for `watch --motion`, plus the small ViT tracker for `track.py`."""
    print('[setup] Motion analysis: installs numpy, opencv-python-headless and scipy (about 150 MB) into '
          f'{motion.MOTION_VENV} and fetches a 0.7 MB object-tracking model.', file=sys.stderr, flush=True)
    if not (_which('ffmpeg') and _which('ffprobe')):
        raise SystemExit('Install FFmpeg first (it provides ffprobe too), then rerun setup.py --install-motion.')
    uv = _ensure_uv()
    venv, python = motion.MOTION_VENV, motion.python_path()
    if not motion.is_ready():
        if venv.is_symlink():
            raise SystemExit('Managed motion venv is a symlink; choose a regular managed directory before installing.')
        if venv.exists():
            shutil.rmtree(venv)  # Only this fixed, installer-owned environment.
        venv.parent.mkdir(parents=True, exist_ok=True)
        _install_step([uv, 'venv', str(venv), '--python', '3.12'], 'Create motion Python 3.12 environment')
        _install_step([uv, 'pip', 'install', '--python', python, *motion.PACKAGES], 'Install motion dependencies')
    check = run_text([str(python), '-I', '-c', 'import cv2, numpy, scipy.signal; '
                      'print(cv2.__version__, numpy.__version__, scipy.__version__)'], timeout=120)
    if check.returncode:
        motion.SENTINEL.unlink(missing_ok=True)
        raise SystemExit(f'Motion environment failed its import check: {diagnostic(check.stderr)}')
    motion.SENTINEL.write_text(check.stdout.strip() + '\n', encoding='utf-8')
    model_ok = _fetch_track_model()
    print(f'[setup] Motion analysis is ready ({check.stdout.strip()}); neural tracker {"ready" if model_ok else "missing (template matching fallback)"}.', file=sys.stderr)
    return 0


def install_tracker():
    """Managed PyTorch venv plus Meta's CoTracker3 (pinned commit, SHA-256-checked weights) for point tracking."""
    print('[setup] Point tracker: installs PyTorch and friends (about 800 MB) into '
          f'{motion.TRACKER_VENV}, fetches Meta\'s CoTracker3 code at a pinned commit from GitHub '
          f'({motion.COTRACKER_REF}) and its 100 MB weights from Hugging Face. CoTracker3 is licensed CC BY-NC 4.0 (non-commercial use).',
          file=sys.stderr, flush=True)
    if not _which('ffmpeg'):
        raise SystemExit('Install FFmpeg first, then rerun setup.py --install-tracker.')
    uv = _ensure_uv()
    venv, python = motion.TRACKER_VENV, motion.tracker_python()
    if not motion.tracker_ready():
        if venv.is_symlink():
            raise SystemExit('Managed tracker venv is a symlink; choose a regular managed directory before installing.')
        if venv.exists():
            shutil.rmtree(venv)  # Only this fixed, installer-owned environment.
        venv.parent.mkdir(parents=True, exist_ok=True)
        _install_step([uv, 'venv', str(venv), '--python', '3.12'], 'Create tracker Python 3.12 environment')
        _install_step([uv, 'pip', 'install', '--python', python, *motion.TRACKER_PACKAGES], 'Install PyTorch and tracker dependencies')
    motion.TRACKER_SENTINEL.unlink(missing_ok=True)
    code = ('import hashlib, os, sys, torch\n'
            f'm = torch.hub.load({motion.COTRACKER_REF!r}, "cotracker3_online", trust_repo=True)\n'
            'p = os.path.join(torch.hub.get_dir(), "checkpoints", "scaled_online.pth")\n'
            'print(hashlib.sha256(open(p, "rb").read()).hexdigest())\n')
    check = run_text([str(python), '-I', '-c', code], timeout=1800)
    digest = (check.stdout.strip().splitlines() or [''])[-1]
    if check.returncode or digest != motion.COTRACKER_WEIGHTS_SHA256:
        raise SystemExit('CoTracker3 did not load or its weights failed the checksum '
                         f'(got {digest[:16] or "nothing"}…): {diagnostic(check.stderr)}')
    motion.TRACKER_SENTINEL.write_text(motion.COTRACKER_REF + '\n', encoding='utf-8')
    print('[setup] CoTracker3 point tracker is ready (weights verified).', file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--json', action='store_true')
    mode.add_argument('--install-whisperx', action='store_true')
    mode.add_argument('--install-motion', action='store_true')
    mode.add_argument('--install-tracker', action='store_true')
    parser.add_argument('--backend', choices=['auto', 'whisperx', 'groq', 'openai', 'none'])
    parser.add_argument('--detail', choices=sorted(DETAILS))
    parser.add_argument('--engine', choices=sorted(ENGINES))
    args = parser.parse_args()
    try:
        if args.check:
            return cmd_check()
        if args.json:
            return cmd_json()
        if args.install_motion:
            return install_motion()
        if args.install_tracker:
            return install_tracker()
        return cmd_install('whisperx' if args.install_whisperx else args.backend, args.detail, args.engine)
    except (ConfigError, OSError, ValueError) as exc:
        print(f'[setup] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    configure_stdio()
    raise SystemExit(main())

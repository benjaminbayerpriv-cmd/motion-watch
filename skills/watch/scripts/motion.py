"""Adapter for the motion analysis engine. Stdlib only; never imports numpy/OpenCV into watch.

The engine (`motion_analyze.py`) runs as a subprocess inside a managed venv, the same way
WhisperX does, so the base skill keeps its no-dependency guarantee.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ENGINE = SCRIPT_DIR / 'motion_analyze.py'
MOTION_VENV = Path(os.environ.get('WATCH_MOTION_VENV') or Path.home() / '.cache' / 'watch' / 'motion-venv')
SENTINEL = MOTION_VENV / '.deps-ok'
MODEL_PATH = Path(os.environ.get('WATCH_TRACK_MODEL') or Path.home() / '.cache' / 'watch' / 'models' / 'vittrack.onnx')
MODEL_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx'
MODEL_SHA256 = '2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d'
TRACKER_VENV = Path(os.environ.get('WATCH_TRACKER_VENV') or Path.home() / '.cache' / 'watch' / 'tracker-venv')
TRACKER_SENTINEL = TRACKER_VENV / '.deps-ok'
TRACKER_PACKAGES = ['torch', 'numpy', 'einops', 'timm', 'tqdm']
COTRACKER_REF = 'facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d'      # pinned commit of Meta's code
COTRACKER_WEIGHTS_URL = 'https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth'
COTRACKER_WEIGHTS_SHA256 = '205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218'
PACKAGES = ['numpy>=1.26,<3', 'opencv-python-headless>=4.9,<6', 'scipy>=1.11,<2']
TRACKER_HINT = 'Run: python3 "{}" --install-tracker'.format(SCRIPT_DIR / 'setup.py')
SETUP_HINT = 'Run: python3 "{}" --install-motion'.format(SCRIPT_DIR / 'setup.py')


def python_path() -> Path:
    return MOTION_VENV / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')


def tracker_python() -> Path:
    return TRACKER_VENV / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')


def tracker_ready() -> bool:
    return TRACKER_SENTINEL.is_file() and tracker_python().is_file()


def is_ready() -> bool:
    return SENTINEL.is_file() and python_path().is_file()


def run(video, out_dir, *, start=None, end=None, max_shots=2, audio=True, view=True, timeout=1800) -> str:
    """Analyse `video`; images and report.json/report.md land in `out_dir`. Returns the markdown report."""
    if not is_ready():
        raise SystemExit(f'Motion analysis needs its helper environment (numpy, OpenCV, SciPy; about 150 MB). {SETUP_HINT}')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python_path()), '-I', str(ENGINE), str(video), '--out', str(out_dir), '--max-shots', str(max_shots)]
    if start:
        cmd += ['--start', f'{start:.3f}']
    if end is not None:
        cmd += ['--end', f'{end:.3f}']
    if not audio:
        cmd.append('--no-audio')
    if not view:
        cmd.append('--no-view')
    try:
        # stderr is inherited so progress shows live; the report is read from disk.
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f'Motion analysis timed out after {timeout}s; narrow it with --start/--end.') from None
    except OSError as exc:
        raise SystemExit(f'Cannot run the motion engine ({type(exc).__name__}). {SETUP_HINT}') from None
    report = out_dir / 'report.md'
    if result.returncode or not report.is_file():
        raise SystemExit(f'Motion analysis failed (exit {result.returncode}); see the messages above.')
    return report.read_text(encoding='utf-8')

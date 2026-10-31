#!/usr/bin/env python3
"""/watch object tools: pick, track, sheet, flow. Stdlib entry point for `motion_track.py`.

    track.py pick  SOURCE --t 12.3 [--at x,y]               frame with a coordinate grid; --at checks a click
    track.py track SOURCE --obj ball@1.2=#3 [--obj ...]     follow objects; path as text, images, video
    track.py sheet SOURCE --start 1 --end 2 --n 8 [--box]   slow-motion contact sheet
    track.py flow  SOURCE --t 12.3                          motion arrows for everything that moves

SOURCE is a local video or a URL (downloaded once into ~/.cache/watch/media and reused).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import auth_args, download, is_url  # noqa: E402
from runtime import configure_stdio  # noqa: E402
import motion  # noqa: E402

ENGINE = SCRIPT_DIR / 'motion_track.py'
MEDIA_CACHE = Path.home() / '.cache' / 'watch' / 'media'
CACHE_DAYS = 3


def prune_cache() -> None:
    """Remove cached downloads untouched for CACHE_DAYS; only ever inside this tool's own cache directory."""
    if not MEDIA_CACHE.is_dir():
        return
    cutoff = time.time() - CACHE_DAYS * 86400
    for entry in MEDIA_CACHE.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry)
        except OSError:
            pass


def resolve_source(source: str, cookies_file=None, cookies_from_browser=None) -> str:
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f'File not found: {path}')
        return str(path)
    auth_args(cookies_file, cookies_from_browser)
    folder = MEDIA_CACHE / hashlib.sha1(source.encode()).hexdigest()[:16]
    cached = sorted(folder.glob('**/video.*')) if folder.is_dir() else []
    cached = [p for p in cached if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov', '.m4v')]
    if cached:
        folder.touch()
        print(f'[track] using cached download {cached[0]}', file=sys.stderr)
        return str(cached[0])
    folder.mkdir(parents=True, exist_ok=True)
    print('[track] downloading media…', file=sys.stderr)
    media = download(source, folder, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser)
    return media['video_path']


def main() -> int:
    ap = argparse.ArgumentParser(prog='track', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('source', help='Local video or URL')
        p.add_argument('--out-dir', default=None, help='Where images and reports go (default: a new temp directory)')
        p.add_argument('--cookies', default=None)
        p.add_argument('--cookies-from-browser', default=None)

    p = sub.add_parser('pick', help='Frame with a coordinate grid; check a click with --at')
    common(p)
    p.add_argument('--t', type=float, required=True, help='Time in seconds')
    p.add_argument('--at', action='append', default=None, help='x,y (0..1) to mark with a cross and a magnified patch; repeatable')
    p.add_argument('--box', default=None, help='x,y,w,h (0..1) to zoom the view on')
    p = sub.add_parser('track', help='Follow one or more objects through the video')
    common(p)
    p.add_argument('--obj', action='append', required=True,
                   help='NAME@TIME=SPEC where SPEC is x,y (the point you click) or x,y,w,h (a box); all 0..1')
    p.add_argument('--size', type=float, default=0.08, help='Region width around a point seed, as a fraction of the frame width (box engine and drawing)')
    p.add_argument('--distance', action='append', default=None, help='a,b: distance between two --obj points over time (mouth opening)')
    p.add_argument('--angle', action='append', default=None, help='a,b,c: angle at b between a and c over time (fin, limb)')
    p.add_argument('--engine', choices=['auto', 'point', 'box'], default='auto',
                   help='point: CoTracker3 follows the clicked point; box: a ViT tracker follows a box around it; auto: point when installed')
    p.add_argument('--start', type=float, default=None, help='Earliest time to track back to')
    p.add_argument('--end', type=float, default=None, help='Latest time to track forward to')
    p.add_argument('--no-video', action='store_true', help='Skip the annotated tracked.mp4 (faster)')
    p.add_argument('--confirm', action='store_true', help='Only show the box that would be tracked, then stop')
    p = sub.add_parser('sheet', help='Contact sheet of frames, optionally zoomed on a box')
    common(p)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--n', type=int, default=8, help='Number of frames (max 24)')
    p.add_argument('--box', default=None, help='x,y,w,h (0..1) to zoom on')
    p.add_argument('--cols', type=int, default=None)
    p.add_argument('--cell', type=int, default=None, help='Cell width in pixels')
    p.add_argument('--grid', action='store_true', help='Overlay a coordinate grid (labels are frame coordinates)')
    p = sub.add_parser('flow', help='Motion arrows and flow colours around a moment')
    common(p)
    p.add_argument('--t', type=float, required=True)
    p.add_argument('--window', type=float, default=0.2, help='Seconds of video the motion is measured over')
    p.add_argument('--raw', action='store_true', help='Keep camera motion instead of removing it')
    args = ap.parse_args()

    if not motion.is_ready():
        raise SystemExit(f'The object tools need the motion environment. {motion.SETUP_HINT}')
    if not motion.MODEL_PATH.is_file():
        print(f'[track] Neural tracker model missing ({motion.MODEL_PATH}); falling back to template matching. '
              f'{motion.SETUP_HINT} fetches it.', file=sys.stderr)
    prune_cache()
    video = resolve_source(args.source, args.cookies, args.cookies_from_browser)
    out = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(tempfile.mkdtemp(prefix=f'watch-{args.cmd}-'))
    out.mkdir(parents=True, exist_ok=True)

    cmd = [str(motion.python_path()), '-I', str(ENGINE), args.cmd, video, '--out', str(out)]
    if args.cmd in ('pick', 'flow'):
        cmd += ['--t', str(args.t)]
    if args.cmd == 'pick':
        for a in (args.at or []):
            cmd += ['--at', a]
        if args.box:
            cmd += ['--box', args.box]
    if args.cmd == 'flow':
        cmd += ['--window', str(args.window)] + (['--raw'] if args.raw else [])
    if args.cmd == 'track':
        for o in args.obj:
            cmd += ['--obj', o]
        cmd += ['--size', str(args.size), '--engine', args.engine]
        for d in (args.distance or []):
            cmd += ['--distance', d]
        for a in (args.angle or []):
            cmd += ['--angle', a]
        if args.start is not None:
            cmd += ['--start', str(args.start)]
        if args.end is not None:
            cmd += ['--end', str(args.end)]
        cmd += (['--no-video'] if args.no_video else []) + (['--confirm'] if args.confirm else [])
    if args.cmd == 'sheet':
        cmd += ['--start', str(args.start), '--end', str(args.end), '--n', str(args.n)]
        for flag, value in (('--box', args.box), ('--cols', args.cols), ('--cell', args.cell)):
            if value is not None:
                cmd += [flag, str(value)]
        cmd += ['--grid'] if args.grid else []
    env = {'WATCH_TRACK_MODEL': str(motion.MODEL_PATH)}
    import os
    result = subprocess.run(cmd, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding='utf-8', errors='replace')
    noise = ('net_impl_backend', 'setPreferableTarget')
    for line in result.stderr.splitlines():
        if not any(n in line for n in noise):
            print(line, file=sys.stderr)
    if result.returncode:
        raise SystemExit(f'{args.cmd} failed (exit {result.returncode}).')
    print(result.stdout)
    print(f'---\n_Output dir: `{out}` — delete when done. Source video: `{video}`._')
    return 0


if __name__ == '__main__':
    configure_stdio()
    raise SystemExit(main())

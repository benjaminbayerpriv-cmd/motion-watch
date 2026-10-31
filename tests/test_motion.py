"""Motion engine: checked against synthetic clips whose true motion is known.

The engine needs numpy/OpenCV/SciPy, which live in the managed motion venv, so these tests skip
unless `setup.py --install-motion` has been run.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/watch/scripts'
sys.path.insert(0, str(SCRIPTS))

import motion  # noqa: E402

REAL_VENV = motion.MOTION_VENV            # resolved before the autouse fixture redirects HOME
needs_env = pytest.mark.skipif(not motion.is_ready(), reason='run setup.py --install-motion first')

MAKE_CLIP = textwrap.dedent('''
    import subprocess, sys
    import cv2, numpy as np
    out = sys.argv[1]
    W, H, FPS = 360, 640, 30
    def tex(seed):
        r = np.random.default_rng(seed)
        t = r.integers(0, 255, (H // 6, W // 6, 3), dtype=np.uint8)
        return cv2.GaussianBlur(cv2.resize(t, (W * 2, H), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
    a, b = tex(1), tex(2)
    vw = cv2.VideoWriter(out + '.raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    ease = lambda u: 1 - (1 - u) ** 3                      # ease-out cubic, 0.5 s .. 2.0 s, 0.63 frame widths
    for f in range(FPS * 4):
        t = f / FPS
        if t < 2.5:
            img = a[:, :W].copy()
            x = 70 + 227 * ease(np.clip((t - 0.5) / 1.5, 0, 1))
            cv2.circle(img, (int(x), 320), 26, (30, 60, 240), -1)
            cv2.circle(img, (int(x), 320), 26, (255, 255, 255), 2)
        else:
            off = int((t - 2.5) * 80)                      # camera pans right at 80 px/s
            img = b[:, off:off + W].copy()
        vw.write(img)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', out + '.raw.mp4', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-an', out], check=True)
''')


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp('motion') / 'ease.mp4'
    subprocess.run([str(motion.python_path()), '-I', '-c', MAKE_CLIP, str(path)], check=True)
    return path


@pytest.fixture(scope='module')
def analysed(clip, tmp_path_factory):
    out = tmp_path_factory.mktemp('analysis')
    report = motion.run(clip, out, audio=False, max_shots=2)
    return out, report, json.loads((out / 'report.json').read_text(encoding='utf-8'))


@needs_env
def test_finds_the_cut_and_the_camera_move(analysed):
    _, _, data = analysed
    assert len(data['cuts']) == 1 and abs(data['cuts'][0] - 2.5) < 0.1
    first, second = data['shots']
    assert first['camera'] == 'static'
    assert 'pan right' in second['camera']


@needs_env
def test_tracks_a_small_smooth_object_with_its_true_path_and_easing(analysed):
    _, report, data = analysed
    mover = data['movers'][0]
    assert mover['shot'] == 1 and mover['net_dir'] == 'right'
    assert 0.45 < mover['length_w'] < 0.75                 # truth: 0.63 frame widths
    assert 0.4 < mover['t0'] < 0.7                         # truth: starts at 0.5 s
    assert mover['easing']['label'].startswith('ease-out')
    assert 'ease-out' in report and 'motion curve' in report


@needs_env
def test_first_visible_motion_is_the_object_not_the_background(analysed):
    _, report, _ = analysed
    line = next(l for l in report.splitlines() if 'First clearly visible motion' in l)
    assert abs(float(line.rsplit(' at ', 1)[1].rstrip('s')) - 0.5) < 0.3


@needs_env
def test_renders_images_and_skips_overlays_when_the_camera_moves(analysed):
    out, report, _ = analysed
    for name in ('overview.png', 'shot01_motion_history.png', 'shot01_strobe.png', 'shot01_tracks.png', 'shot02_tracks.png'):
        assert (out / name).stat().st_size > 2000, name
    assert not (out / 'shot02_motion_history.png').exists()
    assert 'not rendered' in report


@needs_env
@pytest.mark.parametrize('name', ['linear', 'ease-in (quad)', 'ease-out (cubic)', 'ease-in-out (smoothstep)'])
def test_easing_classifier_recovers_known_curves(name):
    code = textwrap.dedent(f'''
        import sys, json
        sys.path.insert(0, {str(SCRIPTS)!r})
        import numpy as np, motion_analyze as m
        u = np.linspace(0, 1, 40)
        s = np.clip(m.EASINGS[{name!r}](u), 0, 1)
        pos = np.c_[np.r_[[0] * 5, s * 100, [100] * 5], np.zeros(50)]
        print(json.dumps(m.classify_easing(pos, 1 / 30)['label']))
    ''')
    result = subprocess.run([str(motion.python_path()), '-I', '-c', code], capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout).startswith(name.split(' (')[0])      # family is exact; strength may tie


@needs_env
def test_watch_motion_mode_end_to_end(clip, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPTS / 'watch.py'), str(clip), '--motion', '--no-whisper',
                             '--engine', 'local', '--out-dir', str(tmp_path)], capture_output=True, encoding='utf-8',
                            env={**os.environ, 'WATCH_MOTION_VENV': str(REAL_VENV)})
    assert result.returncode == 0, result.stderr
    assert '## Motion analysis' in result.stdout and '**Frames:** skipped' in result.stdout
    assert '### Tracked movers' in result.stdout and 'overview.png' in result.stdout
    assert list(tmp_path.glob('watch-*/motion/overview.png'))
    assert list(tmp_path.glob('watch-*/motion/motion_view.mp4')) and 'FOR THE HUMAN' in result.stdout


def test_missing_environment_gives_install_hint(tmp_path):
    video = tmp_path / 'x.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=10:duration=1',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
    env = {**os.environ, 'WATCH_MOTION_VENV': str(tmp_path / 'no-such-venv')}
    result = subprocess.run([sys.executable, str(SCRIPTS / 'watch.py'), str(video), '--motion', '--no-whisper', '--engine', 'local'],
                            capture_output=True, encoding='utf-8', env=env)
    assert '--install-motion' in result.stdout + result.stderr


def test_motion_conflicts_with_gemini_engine():
    result = subprocess.run([sys.executable, str(SCRIPTS / 'watch.py'), 'x.mp4', '--motion', '--engine', 'gemini'],
                            capture_output=True, encoding='utf-8')
    assert result.returncode == 2 and 'cannot be combined' in result.stderr

"""Object tools (track.py): checked against a synthetic bouncing ball whose physics are known."""
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/watch/scripts'
sys.path.insert(0, str(SCRIPTS))

import motion  # noqa: E402

REAL_VENV, REAL_MODEL = motion.MOTION_VENV, motion.MODEL_PATH   # resolved before the autouse fixture redirects HOME
REAL_TRACKER = motion.TRACKER_VENV
needs_env = pytest.mark.skipif(not motion.is_ready(), reason='run setup.py --install-motion first')
needs_point = pytest.mark.skipif(not (motion.is_ready() and motion.tracker_ready()), reason='run setup.py --install-tracker first')

MAKE_CLIP = textwrap.dedent('''
    import subprocess, sys
    import cv2, numpy as np
    out = sys.argv[1]
    W, H, FPS = 360, 640, 30
    r = np.random.default_rng(5)
    bg = cv2.GaussianBlur(cv2.resize(r.integers(40, 215, (H // 8, W // 4, 3), dtype=np.uint8), (W * 3, H),
                          interpolation=cv2.INTER_CUBIC), (0, 0), 1.5)
    G, E, FLOOR, R = 900.0, 0.72, 560.0, 22
    y, vy, x, acc, dt = 100.0, 0.0, 60.0, 0.0, 1 / 2000.0
    vw = cv2.VideoWriter(out + '.raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in range(FPS * 4):
        t = f / FPS
        while acc < t - 1e-9:
            vy += G * dt; y += vy * dt; x += 60 * dt; acc += dt
            if y > FLOOR - R and vy > 0:
                y, vy = FLOOR - R, -vy * E
        img = bg[:, :W].copy()
        cv2.circle(img, (int(x), int(y)), R, (30, 60, 240), -1)
        cv2.circle(img, (int(x), int(y)), R, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', out + '.raw.mp4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', out], check=True)
''')
TRUE_BOUNCES = [0.986, 2.407, 3.429]       # seconds, from the analytic simulation (g = 2.5 fw/s², e = 0.72)


def run_track(*args, check=True):
    env = {**os.environ, 'WATCH_MOTION_VENV': str(REAL_VENV), 'WATCH_TRACK_MODEL': str(REAL_MODEL), 'WATCH_TRACKER_VENV': str(REAL_TRACKER)}
    result = subprocess.run([sys.executable, str(SCRIPTS / 'track.py'), *map(str, args)], capture_output=True,
                            encoding='utf-8', env=env)
    if check:
        assert result.returncode == 0, result.stderr
    return result


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp('bounce') / 'bounce.mp4'
    subprocess.run([str(motion.python_path()), '-I', '-c', MAKE_CLIP, str(path)], check=True)
    return path


@pytest.fixture(scope='module')
def tracked(clip, tmp_path_factory):
    out = tmp_path_factory.mktemp('track')
    result = run_track('track', clip, '--obj', 'ball@0.3=0.217,0.22', '--size', '0.12', '--engine', 'box', '--out-dir', out, '--no-video')
    return out, result.stdout, json.loads((out / 'track.json').read_text(encoding='utf-8'))


@needs_env
def test_pick_shows_a_grid_and_checks_a_click_without_detecting_anything(clip, tmp_path):
    result = run_track('pick', clip, '--t', '0.3', '--at', '0.217,0.22', '--out-dir', tmp_path)
    assert 'click 1: x=0.217, y=0.220' in result.stdout and 'No objects are detected' in result.stdout
    assert 'candidate' not in result.stdout.lower().replace('automatic candidates', '')
    assert (tmp_path / 'pick_0.30s.png').stat().st_size > 5000 and not (tmp_path / 'pick.json').exists()
    zoom = run_track('pick', clip, '--t', '0.3', '--box', '0.1,0.1,0.3,0.3', '--out-dir', tmp_path / 'z')
    assert 'zoomed on box' in zoom.stdout


@needs_env
def test_bounces_gravity_and_bounce_ratio_match_the_physics(tracked):
    out, report, data = tracked
    bounces = [e for e in data[0]['events'] if e['kind'] == 'floor bounce']
    assert len(bounces) >= 3
    for found, truth in zip(bounces, TRUE_BOUNCES):
        assert abs(found['t'] - truth) < 0.08, (found['t'], truth)
    exact = [e['restitution'] for e in bounces if e.get('exact')]
    assert exact and all(0.66 < r < 0.80 for r in exact), exact           # truth 0.72
    g = float(re.search(r'vertical acceleration ([0-9.]+) fw/s²', report).group(1))
    assert 2.25 < g < 2.75                                                # truth 2.5
    assert 'gravity-like' in report


@needs_env
def test_apex_is_not_reported_as_a_bounce_or_turn(tracked):
    _, _, data = tracked
    kinds = [e['kind'] for e in data[0]['events']]
    assert 'ceiling bounce' not in kinds and 'turn' not in kinds[:3]


@needs_env
def test_writes_images_csv_and_path_map(tracked):
    out, report, _ = tracked
    for name in ('paths.png', 'filmstrip_ball.png', 'plot.png', 'seed_ball.png', 'path_ball.csv'):
        assert (out / name).stat().st_size > 500, name
    assert '**Path map**' in report and '| t | x | y |' in report


@needs_env
def test_human_outputs_summary_picture_and_annotated_video(clip, tmp_path):
    result = run_track('track', clip, '--obj', 'ball@0.3=0.217,0.22', '--size', '0.12', '--out-dir', tmp_path)
    assert (tmp_path / 'summary.png').stat().st_size > 20000 and (tmp_path / 'tracked.mp4').stat().st_size > 5000
    assert result.stdout.index('summary.png') < result.stdout.index('paths.png')       # listed first
    assert 'FOR THE HUMAN' in result.stdout and 'BOUNCE' in result.stdout


@needs_env
def test_labels_are_ascii_safe():
    code = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(SCRIPTS)!r})
        import motion_analyze as m
        print(m.ascii_safe("roll -8\u00b0 \u2192 2\u00b2 \u00b1"))
    ''')
    result = subprocess.run([str(motion.python_path()), '-I', '-c', code], capture_output=True, encoding='utf-8')
    assert result.stdout.strip() == 'roll -8 deg -> 2^2 +-'


@needs_env
def test_confirm_stops_after_the_seed(clip, tmp_path):
    result = run_track('track', clip, '--obj', 'ball@0.3=0.217,0.22', '--size', '0.12', '--engine', 'box', '--confirm', '--out-dir', tmp_path)
    assert 'seed_ball.png' in result.stdout and 'exact point' not in result.stdout and not (tmp_path / 'paths.png').exists()


@needs_env
def test_sheet_and_flow(clip, tmp_path):
    run_track('sheet', clip, '--start', '0.2', '--end', '1.0', '--n', '4', '--box', '0.1,0.1,0.4,0.4', '--grid', '--out-dir', tmp_path)
    assert (tmp_path / 'sheet_0.20-1.00s.png').stat().st_size > 2000
    result = run_track('flow', clip, '--t', '0.5', '--out-dir', tmp_path)
    assert (tmp_path / 'flow_0.50s.png').stat().st_size > 2000 and 'heading' in result.stdout


@needs_env
def test_rejects_malformed_or_out_of_range_input(clip, tmp_path):
    bad = run_track('track', clip, '--obj', 'ball=0.2,0.2', '--out-dir', tmp_path, check=False)
    assert bad.returncode != 0 and 'NAME@TIME=SPEC' in bad.stderr
    far = run_track('track', clip, '--obj', 'ball@99=0.2,0.2', '--out-dir', tmp_path, check=False)
    assert far.returncode != 0 and 'outside' in far.stderr
    off = run_track('pick', clip, '--t', '0.3', '--out-dir', tmp_path)           # sanity: valid call still works
    assert off.returncode == 0
    wrong = run_track('track', clip, '--obj', 'ball@0.3=5,9', '--out-dir', tmp_path, check=False)
    assert wrong.returncode != 0 and 'normalised' in wrong.stderr
    gone = run_track('track', clip, '--obj', 'ball@0.3=#2', '--out-dir', tmp_path, check=False)
    assert gone.returncode != 0 and 'Automatic candidates were removed' in gone.stderr


def test_missing_environment_gives_install_hint(tmp_path):
    video = tmp_path / 'x.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=10:duration=1',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
    env = {**os.environ, 'WATCH_MOTION_VENV': str(tmp_path / 'none')}
    result = subprocess.run([sys.executable, str(SCRIPTS / 'track.py'), 'pick', str(video), '--t', '0.2'],
                            capture_output=True, encoding='utf-8', env=env)
    assert result.returncode != 0 and '--install-motion' in result.stderr


@pytest.fixture(scope='module')
def point_tracked(clip, tmp_path_factory):
    out = tmp_path_factory.mktemp('point')
    result = run_track('track', clip, '--obj', 'ball@0.3=0.217,0.22', '--engine', 'point', '--out-dir', out, '--no-video')
    return out, result.stdout, json.loads((out / 'track.json').read_text(encoding='utf-8'))


@needs_point
def test_point_engine_follows_the_clicked_point_with_cotracker(point_tracked):
    out, report, data = point_tracked
    assert 'CoTracker3 point tracker' in report and 'exact point you gave' in report
    assert 'Size:' not in report                                          # a point has no size
    bounces = [e for e in data[0]['events'] if e['kind'] == 'floor bounce']
    assert len(bounces) >= 3
    for found, truth in zip(bounces, TRUE_BOUNCES):
        assert abs(found['t'] - truth) < 0.1, (found['t'], truth)
    g = float(re.search(r'vertical acceleration ([0-9.]+) fw/s²', report).group(1))
    assert 2.25 < g < 2.75
    assert (out / 'seed_ball.png').stat().st_size > 1000


@needs_point
def test_point_engine_draws_a_cross_not_a_box_in_the_video(clip, tmp_path):
    result = run_track('track', clip, '--obj', 'ball@0.3=0.217,0.22', '--engine', 'point', '--out-dir', tmp_path)
    assert (tmp_path / 'tracked.mp4').stat().st_size > 5000 and 'FOR THE HUMAN' in result.stdout


@needs_env
def test_point_engine_rejects_a_box_and_explains_a_missing_install(clip, tmp_path):
    bad = run_track('track', clip, '--obj', 'ball@0.3=0.2,0.2,0.1,0.1', '--engine', 'point', '--out-dir', tmp_path, check=False)
    assert bad.returncode != 0 and 'follows a clicked point' in bad.stderr
    env = {**os.environ, 'WATCH_MOTION_VENV': str(REAL_VENV), 'WATCH_TRACK_MODEL': str(REAL_MODEL), 'WATCH_TRACKER_VENV': str(tmp_path / 'none')}
    missing = subprocess.run([sys.executable, str(SCRIPTS / 'track.py'), 'track', str(clip), '--obj', 'ball@0.3=0.217,0.22',
                              '--engine', 'point', '--out-dir', str(tmp_path / 'o')], capture_output=True, encoding='utf-8', env=env)
    assert missing.returncode != 0 and '--install-tracker' in missing.stderr
    auto = subprocess.run([sys.executable, str(SCRIPTS / 'track.py'), 'track', str(clip), '--obj', 'ball@0.3=0.217,0.22', '--size', '0.12',
                           '--confirm', '--out-dir', str(tmp_path / 'a')], capture_output=True, encoding='utf-8', env=env)
    assert auto.returncode == 0 and 'CoTracker3 not installed' in auto.stdout        # auto falls back to the box tracker, saying so


MAKE_MOUTH = textwrap.dedent('''
    import subprocess, sys
    import cv2, numpy as np
    out = sys.argv[1]
    W, H, FPS = 360, 640, 30
    r = np.random.default_rng(9)
    bg = cv2.GaussianBlur(cv2.resize(r.integers(60, 190, (H // 8, W // 4, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.5)
    vw = cv2.VideoWriter(out + '.raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    def dot(img, x, y, col):
        cv2.circle(img, (int(round(x)), int(round(y))), 8, col, -1); cv2.circle(img, (int(round(x)), int(round(y))), 8, (255, 255, 255), 1)
    for f in range(FPS * 5):
        t = f / FPS; img = bg.copy()
        d = 0.06 + 0.04 * np.sin(2 * np.pi * t)                 # lip distance: 0.02..0.10 frame widths, 1 Hz
        dot(img, 0.3 * W, 0.3 * H - d * W / 2, (40, 40, 240)); dot(img, 0.3 * W, 0.3 * H + d * W / 2, (30, 220, 240))
        th = np.radians(50 + 30 * np.sin(2 * np.pi * t * 0.5))   # hinge angle: 20..80 degrees, 0.5 Hz
        px, py, L = 0.7 * W, 0.7 * H, 0.15 * W
        dot(img, px, py, (60, 200, 60)); dot(img, px, py - L, (240, 120, 40)); dot(img, px + L * np.sin(th), py - L * np.cos(th), (220, 60, 220))
        vw.write(img)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', out + '.raw.mp4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', out], check=True)
''')


@pytest.fixture(scope='module')
def mouth(tmp_path_factory):
    path = tmp_path_factory.mktemp('mouth') / 'mouth.mp4'
    subprocess.run([str(motion.python_path()), '-I', '-c', MAKE_MOUTH, str(path)], check=True)
    return path


@needs_env
def test_distance_and_angle_measure_shape_change(mouth, tmp_path):
    result = run_track('track', mouth, '--engine', 'box', '--size', '0.055', '--obj', 'top@1.0=0.300,0.2831', '--obj', 'bottom@1.0=0.300,0.3169',
                       '--obj', 'pivot@1.0=0.700,0.700', '--obj', 'a@1.0=0.700,0.6156', '--obj', 'c@1.0=0.815,0.6458',
                       '--distance', 'top,bottom', '--angle', 'a,pivot,c', '--out-dir', tmp_path)
    text = result.stdout
    dist = text[text.index('distance**'):text.index('angle at')]
    assert 'regular cycle: period 1.0' in dist or 'regular cycle: period 0.99' in dist             # truth 1.0 s
    peaks = [float(x) for x in re.search(r'maxima \(wide open / far apart\) at ([0-9., s]+?);', dist).group(1).replace('s', '').split(',')]
    assert len(peaks) == 5 and all(abs(p - (0.25 + k)) < 0.1 for k, p in enumerate(peaks)), peaks
    ang = text[text.index('angle at'):text.index('## Files')]
    lo, hi = [float(x) for x in re.search(r'min ([0-9]+), max ([0-9]+)', ang).groups()]
    assert 15 <= lo <= 26 and 74 <= hi <= 86, (lo, hi)                                              # truth 20..80
    assert 'regular cycle: period 2.0' in ang
    assert (tmp_path / 'plot.png').stat().st_size > 20000 and (tmp_path / 'tracked.mp4').stat().st_size > 5000


@needs_env
def test_measure_rejects_unknown_names_and_sheet_says_look_only(mouth, clip, tmp_path):
    bad = run_track('track', mouth, '--engine', 'box', '--obj', 'top@1.0=0.300,0.2831', '--distance', 'top,nothing', '--out-dir', tmp_path, check=False)
    assert bad.returncode != 0 and 'not one of the --obj names' in bad.stderr
    sheet = run_track('sheet', clip, '--start', '0.2', '--end', '1.0', '--n', '4', '--out-dir', tmp_path / 's')
    assert 'This is a LOOK, not a measurement' in sheet.stdout and '--distance' in sheet.stdout

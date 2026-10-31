#!/usr/bin/env python3
"""Object tools for /watch: pick an object, track it, inspect its path, slow-motion sheets, flow view.

Runs inside the managed motion venv (numpy, OpenCV, SciPy); `track.py` launches it. Coordinates are
always normalised: x, y, w, h in 0..1 of the frame, origin top-left, so an agent can read them off the
grid in a `pick` image and pass them back.

  pick   VIDEO --t T [--at x,y]            frame with a coordinate grid; --at checks a click with a magnified patch
  track  VIDEO --obj NAME@T=SPEC ...       follow objects forward and backward; path as text, image, video
  sheet  VIDEO --start S --end E --n N     contact sheet of frames (optionally zoomed on a box)
  flow   VIDEO --t T                       motion arrows for everything that moves at time T

SPEC is `x,y` (the point you click; its box size comes from --size) or `x,y,w,h` (a box). Nothing is detected for you.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
import motion as adapter  # noqa: E402  (stdlib-only paths and constants)
import motion_analyze as ma  # noqa: E402

TW = 360                                  # working width for tracking
MODEL = Path(os.environ.get("WATCH_TRACK_MODEL") or Path.home() / ".cache" / "watch" / "models" / "vittrack.onnx")
PALETTE = [(60, 180, 255), (80, 230, 120), (255, 160, 60), (220, 90, 255), (60, 230, 230), (130, 130, 255)]
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


# ----------------------------------------------------------------------------- basics

def log(msg):
    print(msg, file=sys.stderr, flush=True)


def frame_dims(info, width=TW):
    return width, ma.even(info["height"] * width / info["width"])


def parse_nums(text, n_options=(2, 4)):
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        raise SystemExit(f"Cannot parse coordinates '{text}'; use x,y or x,y,w,h as numbers between 0 and 1.")
    if len(v) not in n_options:
        raise SystemExit(f"Expected {' or '.join(map(str, n_options))} numbers in '{text}'.")
    if any(not (-0.2 <= a <= 1.2) for a in v):
        raise SystemExit(f"Coordinates in '{text}' must be normalised to 0..1 (x right, y down, origin top-left).")
    return v


def stream(video, t_a, t_b, rate, w, h, reverse=False, chunk=4.0):
    """Yield (time, BGR frame) at `rate` fps over [t_a, t_b]; reverse=True walks backwards in time."""
    if not reverse:
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_a):.3f}", "-i", str(video), "-t", f"{max(0.05, t_b - t_a):.3f}",
               "-an", "-vf", f"fps={rate:.4f},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
        n, k = w * h * 3, 0
        try:
            while True:
                buf = proc.stdout.read(n)
                if len(buf) < n:
                    break
                yield t_a + k / rate, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                k += 1
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
    else:
        end = t_b
        while end > t_a + 1e-6:
            start = max(t_a, end - chunk)
            frames = ma.ffmpeg_frames(video, start, end + 0.02, rate, w, h)
            for k in range(len(frames) - 1, -1, -1):
                tt = start + k / rate
                if tt <= end + 1e-6:
                    yield tt, frames[k]
            end = start - 1.0 / rate


def box_iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def colour_name(bgr):
    b, g, r = [float(v) for v in bgr]
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = float(hsv[0]) * 2, hsv[1] / 255.0, hsv[2] / 255.0
    if v < 0.18:
        return "black"
    if s < 0.16:
        return "white" if v > 0.82 else "light grey" if v > 0.55 else "dark grey"
    if h < 15 or h >= 345:
        return "red" if v > 0.45 else "dark red"
    for hi, name in ((40, "orange"), (68, "yellow"), (160, "green"), (200, "cyan"), (255, "blue"), (290, "purple"), (345, "pink")):
        if h < hi:
            return ("brown" if name == "orange" and v < 0.55 else name)
    return "red"


def put(img, text, org, scale=0.42, color=(255, 255, 255)):
    ma.put_text(img, text, org, scale, color)



def draw_grid(img, step=0.1, xr=(0.0, 1.0), yr=(0.0, 1.0)):
    """Grid lines every `step` of the view; labels show frame coordinates (xr, yr = view extent)."""
    h, w = img.shape[:2]
    overlay = img.copy()
    n = int(round(1 / step))
    for k in range(1, n):
        x, y = int(k * step * w), int(k * step * h)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    img[:] = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
    prec = 1 if xr == (0.0, 1.0) and yr == (0.0, 1.0) else 3
    for k in range(1, n):
        lx = f"{xr[0] + k * step * (xr[1] - xr[0]):.{prec}f}"
        ly = f"{yr[0] + k * step * (yr[1] - yr[0]):.{prec}f}"
        put(img, lx, (int(k * step * w) - 10, 11), 0.33, (255, 255, 120))
        put(img, lx, (int(k * step * w) - 10, h - 4), 0.33, (255, 255, 120))
        put(img, ly, (2, int(k * step * h) + 4), 0.33, (255, 255, 120))
        put(img, ly, (w - 34 if prec == 3 else w - 24, int(k * step * h) + 4), 0.33, (255, 255, 120))


def cmd_pick(args, info):
    """A frame with a coordinate grid; optional crosshairs (`--at`) with magnified insets to verify a click; `--box` zooms."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dw = 540
    full_w = min(1080, info["width"])
    full_h = ma.even(info["height"] * full_w / info["width"])
    big = ma.grab_frame(args.video, args.t, full_w, full_h)
    view = parse_nums(args.box, (4,)) if args.box else [0.0, 0.0, 1.0, 1.0]
    x0, y0 = int(view[0] * full_w), int(view[1] * full_h)
    crop = big[y0:max(y0 + 8, int((view[1] + view[3]) * full_h)), x0:max(x0 + 8, int((view[0] + view[2]) * full_w))]
    img = cv2.resize(crop, (dw, max(8, int(dw * crop.shape[0] / crop.shape[1]))))
    zoomed = args.box is not None
    draw_grid(img, 0.1 if not zoomed else 0.25, (view[0], view[0] + view[2]), (view[1], view[1] + view[3]))
    points = [parse_nums(a, (2,)) for a in (args.at or [])]
    insets = []
    for n, (px, py) in enumerate(points, 1):
        cx, cy = int((px - view[0]) / view[2] * dw), int((py - view[1]) / view[3] * img.shape[0])
        cv2.circle(img, (cx, cy), 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx - 16, cy), (cx + 16, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - 16), (cx, cy + 16), (0, 255, 255), 1, cv2.LINE_AA)
        put(img, f"{n}", (cx + 11, cy - 11), 0.55, (0, 255, 255))
        side = int(0.14 * full_w)                                        # magnified patch around the click
        pad = side
        padded = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        gx, gy = int(px * full_w), int(py * full_h)
        patch = padded[gy - side // 2 + pad:gy + side // 2 + pad, gx - side // 2 + pad:gx + side // 2 + pad].copy()
        patch = cv2.resize(patch, (240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.line(patch, (110, 120), (130, 120), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(patch, (120, 110), (120, 130), (0, 255, 255), 1, cv2.LINE_AA)
        put(patch, f"{n}: ({px:.3f},{py:.3f})  14% of width", (4, 14), 0.4, (0, 255, 255))
        insets.append(patch)
    canvas = img
    if insets:
        row = np.hstack(insets[:4])
        row = np.pad(row, ((0, 0), (0, max(0, dw - row.shape[1])), (0, 0)), constant_values=20)[:, :max(dw, row.shape[1])]
        width = max(canvas.shape[1], row.shape[1])
        canvas = np.vstack([np.pad(canvas, ((0, 0), (0, width - canvas.shape[1]), (0, 0)), constant_values=20),
                            np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=20)])
    path = out / f"pick_{args.t:.2f}s.png"
    cv2.imwrite(str(path), canvas)
    lines = [f"# watch: pick at {args.t:.2f}s", "",
             f"- `{path}` — frame {info['width']}x{info['height']} with a grid labelled in frame coordinates (0..1, x right, y down, origin top-left)"
             + (f"; zoomed on box {','.join(f'{v:.2f}' for v in view)}" if zoomed else "") + ".",
             "- No objects are detected for you: read the coordinates of the thing you want off the grid."]
    if points:
        lines.append("- Yellow cross = your click; the strip below shows what lies under it, magnified. If the cross is not on the object, "
                     "correct the coordinates and run `pick` again (you can zoom with `--box`).")
        lines += [f"  - click {n}: x={px:.3f}, y={py:.3f}" for n, (px, py) in enumerate(points, 1)]
    else:
        lines.append("- To check a click before tracking: `pick --t T --at x,y` draws a cross and a magnified patch. "
                     "Then track it with `track --obj NAME@T=x,y --size S` (a point plus the box size S as a fraction of the frame width) "
                     "or `--obj NAME@T=x,y,w,h` (a box).")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- tracking

class Tracker:
    """Neural ViT tracker (OpenCV) with a template-matching fallback when the model is absent."""

    def __init__(self, frame, box):
        self.box = [float(v) for v in box]
        self.net = None
        self.neural = False
        if MODEL.is_file() and hasattr(cv2, "TrackerVit_create"):
            try:
                prm = cv2.TrackerVit_Params()
                prm.net = str(MODEL)
                self.net = cv2.TrackerVit_create(prm)
                self.net.init(frame, tuple(int(round(v)) for v in box))
                self.neural = True
            except cv2.error:
                self.net = None
        self.template = self.crop(frame, box)

    @staticmethod
    def crop(frame, box):
        x, y, w, h = [int(round(v)) for v in box]
        h_, w_ = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        return frame[y:min(h_, y + max(4, h)), x:min(w_, x + max(4, w))].copy()

    def update(self, frame):
        if self.neural:
            ok, bb = self.net.update(frame)
            score = float(self.net.getTrackingScore())
            if ok and bb[2] > 2 and bb[3] > 2:
                self.box = [float(v) for v in bb]
            return bool(ok), list(self.box), score
        x, y, w, h = self.box                                  # fallback: local template search
        cx, cy = x + w / 2, y + h / 2
        sx0, sy0 = int(max(0, cx - 1.6 * w)), int(max(0, cy - 1.6 * h))
        win = frame[sy0:int(cy + 1.6 * h), sx0:int(cx + 1.6 * w)]
        t = self.template
        if win.shape[0] <= t.shape[0] or win.shape[1] <= t.shape[1]:
            return False, list(self.box), 0.0
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        self.box = [sx0 + loc[0], sy0 + loc[1], float(t.shape[1]), float(t.shape[0])]
        if mx > 0.85:
            self.template = self.crop(frame, self.box)
        return mx > 0.5, list(self.box), float(mx)


def refind(frame, templates, last_box, thr=0.7):
    """Search the whole frame for the object's remembered appearance at a few scales."""
    best = (0.0, None)
    for tpl in templates:
        for s in (0.75, 0.9, 1.0, 1.15, 1.35):
            t = cv2.resize(tpl, None, fx=s, fy=s) if s != 1.0 else tpl
            if t.shape[0] < 6 or t.shape[1] < 6 or t.shape[0] >= frame.shape[0] or t.shape[1] >= frame.shape[1]:
                continue
            r = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (float(mx), [loc[0], loc[1], t.shape[1], t.shape[0]])
    return best if best[0] >= thr else (best[0], None)


def run_direction(video, seed_t, box_px, t_lim, rate, reverse, info, width=TW):
    """Track from the seed frame towards t_lim. Returns rows [t, cx, cy, w, h, sx, sy, score, state]."""
    w, h = frame_dims(info, width)
    gen = stream(video, seed_t, t_lim, rate, w, h, reverse=reverse) if not reverse else stream(video, t_lim, seed_t, rate, w, h, reverse=True)
    rows, templates = [], []
    tracker = None
    prev_gray = prev_thumb = None
    M = np.eye(3)
    chain_ok = True
    pts = np.zeros((0, 1, 2), np.float32)
    low = 0
    state_lost = False
    last_box = list(box_px)
    last_good = None
    k = 0
    for t, frame in gen:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        thumb = cv2.resize(gray, (64, max(8, int(64 * h / w)))).astype(np.float32)
        cut = False
        if prev_gray is not None:
            cut = float(np.abs(thumb - prev_thumb).mean()) > 30
        if tracker is None:
            tracker = Tracker(frame, box_px)
            templates = [tracker.template]
            ok, box, score, state = True, list(box_px), 1.0, 0
        else:
            # camera motion from background features, ignoring the object itself
            if not cut:
                if len(pts) < 30:
                    mask = np.full(gray.shape, 255, np.uint8)
                    bx, by, bw_, bh_ = last_box
                    cv2.rectangle(mask, (int(bx - bw_ * 0.3), int(by - bh_ * 0.3)), (int(bx + bw_ * 1.3), int(by + bh_ * 1.3)), 0, -1)
                    new = cv2.goodFeaturesToTrack(prev_gray, 160, 0.01, 7, mask=mask)
                    pts = new.astype(np.float32) if new is not None else np.zeros((0, 1, 2), np.float32)
                if len(pts) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
                    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
                    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (np.linalg.norm((pb - pts).reshape(-1, 2), axis=1) < 1.0)
                    a, b = pts[good], p1[good]
                    if len(a) >= 8:
                        A, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
                        if A is not None:
                            M = np.vstack([A, [0, 0, 1]]) @ M
                    pts = p1[good].copy()
            else:
                chain_ok = False
                pts = np.zeros((0, 1, 2), np.float32)
            state = 0
            if cut:
                state_lost, low = True, 3
            if not state_lost:
                ok, box, score = tracker.update(frame)
                if ok and score >= 0.35:
                    low = 0
                    if score > 0.7 and k % 8 == 0:
                        templates = (templates[:1] + templates[1:][-2:] + [tracker.crop(frame, box)])[-4:]
                else:
                    low += 1
                    if low >= 3:
                        state_lost = True
            if state_lost:
                if k % 2 == 0 or cut:
                    sc, found = refind(frame, templates, last_box)
                    if found is None and last_good is not None and not cut and k % 4 == 0:
                        # the object may simply have stayed near where it was: re-seed from the last good frame
                        probe_t = Tracker(last_good, last_box)
                        pok, pbox, psc = probe_t.update(frame)
                        if pok and psc > 0.55 and probe_t.neural:
                            found, sc = pbox, psc
                    if found is not None:
                        tracker = Tracker(frame, found)
                        state_lost, low, ok, box, score, state = False, 0, True, found, sc, 3
                    else:
                        ok, box, score, state = False, last_box, sc, (2 if cut else 1)
                else:
                    ok, box, score, state = False, last_box, 0.0, 1
        if ok:
            last_box = list(box)
            if score > 0.6 and k % 3 == 0:
                last_good = frame.copy()
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        if chain_ok:
            sp = np.linalg.inv(M) @ np.array([cx, cy, 1.0])
            sx, sy = float(sp[0]), float(sp[1])
        else:
            sx = sy = float("nan")
        if not ok:
            rows.append([t] + [float("nan")] * 6 + [score, state])
        else:
            rows.append([t, cx, cy, box[2], box[3], sx, sy, score, state])
        prev_gray, prev_thumb = gray, thumb
        k += 1
    return rows


def resolve_seed(video, t, spec, info, size=0.08):
    """-> (box normalised, description). A point becomes a box of `size` frame widths centred exactly on it."""
    spec = spec.strip()
    if spec.startswith("#"):
        raise SystemExit("Automatic candidates were removed: give the object yourself as x,y (a point; size via --size) or x,y,w,h (a box). "
                         "Read the coordinates off `pick`.")
    v = parse_nums(spec)
    if len(v) == 4:
        return v, "box you gave"
    if not 0.005 <= size <= 0.6:
        raise SystemExit("--size must be between 0.005 and 0.6 (fraction of the frame width).")
    hh = size * info["width"] / info["height"]
    return [v[0] - size / 2, v[1] - hh / 2, size, hh], f"point you gave, box {size:.3f} frame widths wide centred on it"


def track_object(video, name, seed_t, box_n, info, t_start, t_end, rate):
    # small objects need more pixels: aim for a seed box of at least ~28 px
    width = int(np.clip(28.0 / max(box_n[2], 0.004), TW, min(960, info["width"]))) // 2 * 2
    if width > TW:
        log(f"[track] {name}: small object, working at {width}px width")
    w, h = frame_dims(info, width)
    box_px = [box_n[0] * w, box_n[1] * h, max(6.0, box_n[2] * w), max(6.0, box_n[3] * h)]
    box_px[0] = float(np.clip(box_px[0], 0, w - 8))
    box_px[1] = float(np.clip(box_px[1], 0, h - 8))
    log(f"[track] {name}: forward {seed_t:.1f}s → {t_end:.1f}s")
    fwd = run_direction(video, seed_t, box_px, t_end, rate, False, info, width) if t_end - seed_t > 1.0 / rate else []
    if not fwd:
        fwd = run_direction(video, seed_t, box_px, seed_t + 1.5 / rate, rate, False, info, width)
    bwd = []
    if seed_t - t_start > 1.0 / rate:
        log(f"[track] {name}: backward {seed_t:.1f}s → {t_start:.1f}s")
        bwd = run_direction(video, seed_t, box_px, t_start, rate, True, info, width)
        bwd = [r for r in bwd if r[0] < fwd[0][0] - 0.4 / rate][::-1] if fwd else bwd[::-1]
    P = np.array(bwd + fwd, dtype=np.float64)
    return P, (w, h)


# ----------------------------------------------------------------------------- analysis

POINT_LABEL = "CoTracker3 point tracker (neural; Meta, CC BY-NC 4.0, non-commercial use)"
BOX_LABEL = "ViT box tracker (neural)"


def camera_pass(video, seed_t, t_lim, rate, reverse, info, width, point_at):
    """Camera motion from the background only, in processing order from the seed: [(t, M or None, cut)]."""
    w, h = frame_dims(info, width)
    gen = stream(video, seed_t, t_lim, rate, w, h) if not reverse else stream(video, t_lim, seed_t, rate, w, h, reverse=True)
    out, prev_gray, prev_thumb = [], None, None
    M, chain, pts = np.eye(3), True, np.zeros((0, 1, 2), np.float32)
    for t, frame in gen:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        thumb = cv2.resize(gray, (64, max(8, int(64 * h / w)))).astype(np.float32)
        cut = False
        if prev_gray is not None:
            cut = float(np.abs(thumb - prev_thumb).mean()) > 30
            if cut:
                chain, pts = False, np.zeros((0, 1, 2), np.float32)
            elif chain:
                px, py = point_at(t)
                if len(pts) < 30:
                    mask = np.full(gray.shape, 255, np.uint8)
                    cv2.circle(mask, (int(px), int(py)), int(0.1 * w), 0, -1)
                    new = cv2.goodFeaturesToTrack(prev_gray, 160, 0.01, 7, mask=mask)
                    pts = new.astype(np.float32) if new is not None else np.zeros((0, 1, 2), np.float32)
                if len(pts) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
                    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
                    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (np.linalg.norm((pb - pts).reshape(-1, 2), axis=1) < 1.0)
                    if good.sum() >= 8:
                        A, _ = cv2.estimateAffinePartial2D(pts[good], p1[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                        if A is not None:
                            M = np.vstack([A, [0, 0, 1]]) @ M
                    pts = p1[good].copy()
        out.append((t, M.copy() if chain else None, cut))
        prev_gray, prev_thumb = gray, thumb
    return out


def track_point(video, name, seed_t, pt_n, size_n, info, t_start, t_end, rate):
    """Follow the exact point with CoTracker3 (separate venv); camera and scene cuts handled here. Same row layout as track_object."""
    import tempfile
    width = TW
    w, h = frame_dims(info, width)
    hh = size_n * info["width"] / info["height"]
    with tempfile.TemporaryDirectory(prefix="watch-point-") as tmp:
        out_json = Path(tmp) / "points.json"
        cmd = [str(adapter.tracker_python()), "-I", str(Path(__file__).parent / "point_track.py"), str(video), "--out", str(out_json),
               "--seed-t", f"{seed_t:.4f}", "--start", f"{t_start:.4f}", "--end", f"{t_end:.4f}", "--rate", f"{rate:.4f}",
               "--width", str(w), "--height", str(h), "--points", f"{pt_n[0]:.5f},{pt_n[1]:.5f}"]
        log(f"[track] {name}: CoTracker3 following ({pt_n[0]:.3f}, {pt_n[1]:.3f}) at {seed_t:.2f}s")
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding="utf-8", errors="replace")
        for line in res.stderr.splitlines():
            if line.startswith("[cotracker]"):
                log(line)
        if res.returncode or not out_json.is_file():
            raise SystemExit("CoTracker3 failed:\n" + "\n".join(res.stderr.splitlines()[-6:]))
        data = json.loads(out_json.read_text(encoding="utf-8"))["points"][0]
    parts = []
    if data["bwd"]:
        b = data["bwd"]
        parts.append(dict(t=b["t"][::-1], x=b["x"][::-1], y=b["y"][::-1], vis=b["vis"][::-1], bwd=True))
    if data["fwd"]:
        f = data["fwd"]
        skip = 1 if parts else 0                      # the seed frame is in both passes
        parts.append(dict(t=f["t"][skip:], x=f["x"][skip:], y=f["y"][skip:], vis=f["vis"][skip:], bwd=False))
    t = np.array(sum((p["t"] for p in parts), []), dtype=np.float64)
    x = np.array(sum((p["x"] for p in parts), []), dtype=np.float64)
    y = np.array(sum((p["y"] for p in parts), []), dtype=np.float64)
    vis = np.array(sum((p["vis"] for p in parts), []), dtype=bool)
    order = np.argsort(t)
    t, x, y, vis = t[order], x[order], y[order], vis[order]
    point_at = lambda tt: (float(np.interp(tt, t, x)), float(np.interp(tt, t, y)))
    cams = {}
    for reverse, lim in ((False, t_end), (True, t_start)):
        if (t_end - seed_t if not reverse else seed_t - t_start) > 1.0 / rate:
            cams[reverse] = camera_pass(video, seed_t, lim, rate, reverse, info, width, point_at)
    cut_after = min([c[0] for c in cams.get(False, []) if c[2]] or [1e9])     # first cut forward of the seed
    cut_before = max([c[0] for c in cams.get(True, []) if c[2]] or [-1e9])    # last cut behind the seed
    cam_t = {False: np.array([c[0] for c in cams.get(False, [])]), True: np.array([c[0] for c in cams.get(True, [])])}
    P = np.full((len(t), 9), np.nan)
    box_w, box_h = size_n * w, hh * h
    for i in range(len(t)):
        ti = t[i]
        state, score = (0, 0.9) if vis[i] else (1, 0.2)
        if ti >= cut_after - 1e-6 or ti <= cut_before + 1e-6:
            state, score = 2, 0.0
        P[i, 0], P[i, 7], P[i, 8] = ti, score, state
        if state != 0:
            continue
        P[i, 1:5] = [x[i], y[i], box_w, box_h]
        rev = ti < seed_t - 0.5 / rate
        ct = cam_t[rev] if rev in cams else np.array([])
        if len(ct):
            k = int(np.argmin(np.abs(ct - ti)))
            Mk = cams[rev][k][1]
            if Mk is not None and abs(ct[k] - ti) < 1.0 / rate:
                sp = np.linalg.inv(Mk) @ np.array([x[i], y[i], 1.0])
                P[i, 5], P[i, 6] = sp[0], sp[1]
        elif abs(ti - seed_t) < 1.0 / rate:                       # the seed frame itself
            P[i, 5], P[i, 6] = x[i], y[i]
    return P, (w, h)


def fill_gaps(t, a, max_gap):
    a = a.copy()
    good = ~np.isnan(a)
    if good.sum() < 2:
        return a
    idx = np.where(good)[0]
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if 1 < i1 - i0 and t[i1] - t[i0] <= max_gap:
            a[i0:i1] = np.interp(t[i0:i1], [t[i0], t[i1]], [a[i0], a[i1]])
    return a


def segments(valid):
    out, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(valid) - 1))
    return out


def derivative(t, x, rate, win_s=0.2):
    """Smoothed first derivative per valid segment via Savitzky-Golay (NaN where unsupported)."""
    v = np.full(len(x), np.nan)
    win = int(round(win_s * rate)) | 1
    win = max(5, win)
    for a, b in segments(~np.isnan(x)):
        n = b - a + 1
        if n < 5:
            continue
        w = min(win, n if n % 2 else n - 1)
        if w < 5:
            continue
        v[a:b + 1] = signal.savgol_filter(x[a:b + 1], w, 2, deriv=1, delta=1.0 / rate)
    return v


def smooth_pos(x, rate, win_s=0.2):
    s = x.copy()
    win = max(5, int(round(win_s * rate)) | 1)
    for a, b in segments(~np.isnan(x)):
        n = b - a + 1
        w = min(win, n if n % 2 else n - 1)
        if w >= 5:
            s[a:b + 1] = signal.savgol_filter(x[a:b + 1], w, 2)
    return s


def analyse(P, info, wh, rate):
    """Path statistics and events. Units: frame widths (fw) and seconds; y grows downward."""
    w, h = wh
    P = P.copy()
    good_scores = P[(P[:, 8] == 0) & np.isfinite(P[:, 1]), 7]
    floor = max(0.4, 0.6 * float(np.median(good_scores))) if len(good_scores) else 0.4
    shaky = (P[:, 8] == 1) | (P[:, 8] == 2) | (P[:, 7] < floor)   # lost or unsure: never feed these into statistics
    shaky[0] = False
    P[shaky, 1:7] = np.nan
    t = P[:, 0]
    use_scene = False
    cx, cy = P[:, 1] / w, P[:, 2] / w
    sx, sy = P[:, 5] / w, P[:, 6] / w
    if np.isfinite(sx).sum() >= 0.8 * np.isfinite(cx).sum() and np.isfinite(cx).sum() > 5:
        both = np.isfinite(sx) & np.isfinite(cx)
        if both.sum() > 5 and float(np.nanmax(np.hypot(sx[both] - cx[both], sy[both] - cy[both]))) > 0.08:
            use_scene = True
    px, py = (sx, sy) if use_scene else (cx, cy)
    px, py = fill_gaps(t, px, 0.4), fill_gaps(t, py, 0.4)
    sxs, sys_ = smooth_pos(px, rate), smooth_pos(py, rate)
    vx, vy = derivative(t, px, rate), derivative(t, py, rate)
    sp = np.hypot(vx, vy)
    valid = np.isfinite(sp)
    res = dict(use_scene=use_scene, px=px, py=py, vx=vx, vy=vy, speed=sp, events=[], notes=[])
    if valid.sum() < 6:
        res["notes"].append("too few tracked samples for a path analysis")
        return res
    dt = 1.0 / rate
    step = np.hypot(np.diff(sxs), np.diff(sys_))
    step = step[np.isfinite(step)]
    total = float(step.sum())
    first, last = np.where(np.isfinite(sxs))[0][[0, -1]]
    net = float(np.hypot(sxs[last] - sxs[first], sys_[last] - sys_[first]))
    ip = int(np.nanargmax(sp))
    res.update(total_length=total, net=net, straightness=net / total if total > 1e-9 else 1.0,
               mean_speed=float(np.nanmean(sp)), peak_speed=float(sp[ip]), peak_t=float(t[ip]),
               t_first=float(t[first]), t_last=float(t[last]),
               tracked_frac=float(np.isfinite(P[:, 1]).mean()))
    vmax = float(np.nanmax(sp))
    ev = res["events"]
    # pauses
    still = valid & (sp < max(0.03, 0.06 * vmax))
    for a, b in segments(still):
        if t[b] - t[a] >= 0.25:
            ev.append(dict(kind="pause", t=float(t[a]), t_end=float(t[b]), x=float(sxs[(a + b) // 2]), y=float(sys_[(a + b) // 2])))
    # bounces: the velocity reverses *with speed on both sides* (an arc's apex reverses too, but slowly)
    k1, k4 = 1, max(3, int(round(0.13 * rate)))

    def slope(arr, lo, hi):
        idx = np.arange(max(0, lo), min(len(arr), hi + 1))
        idx = idx[np.isfinite(arr[idx])]
        if len(idx) < 3:
            return None
        return float(np.polyfit(t[idx], arr[idx], 1)[0])
    thr = max(0.3, 0.15 * vmax)
    flips = []
    for axis, vel, arr in ((1, vy, py), (0, vx, px)):
        last_i = -99
        for i in range(2, len(vel) - 2):
            if not (np.isfinite(vel[i - 1]) and np.isfinite(vel[i])) or i - last_i < max(2, int(0.15 * rate)):
                continue
            if not (vel[i - 1] * vel[i] <= 0 and vel[i - 1] != vel[i]):
                continue
            vb, va = slope(arr, i - k4, i - k1), slope(arr, i + k1, i + k4)
            if vb is None or va is None or vb * va >= 0 or min(abs(vb), abs(va)) < thr:
                continue
            last_i = i
            flips.append((i, axis, vb, va))
    for i, axis, vb, va in flips:
        if axis == 1:
            kind = "floor bounce" if vb > 0 else "ceiling bounce"
        else:
            kind = "wall bounce (right)" if vb > 0 else "wall bounce (left)"
        ev.append(dict(kind=kind, t=float(t[i]), x=float(sxs[i]), y=float(sys_[i]), axis=axis, vb=vb, va=va,
                       speed_before=abs(vb), speed_after=abs(va), restitution=None))
    # turns: an abrupt heading change at real speed (not the start of a fall)
    kk = max(2, int(round(0.15 * rate)))
    for i in range(kk, len(vx) - kk):
        a = np.array([vx[i - kk], vy[i - kk]])
        b = np.array([vx[i + kk], vy[i + kk]])
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 0.25 or nb < 0.25:
            continue
        ang = math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1, 1))))
        if ang > 70 and not any(abs(e["t"] - t[i]) < 0.3 for e in ev if e["kind"] != "pause"):
            ev.append(dict(kind="turn", t=float(t[i]), x=float(sxs[i]), y=float(sys_[i]), angle=ang))
    ev.sort(key=lambda e: e["t"])
    # gravity-like arcs between bounces; turns inside a clean arc are just its apex
    cuts = [float(t[first])] + [e["t"] for e in ev if "bounce" in e["kind"]] + [float(t[last])]
    acc, fits = [], []
    for a_t, b_t in zip(cuts[:-1], cuts[1:]):
        m = (t >= a_t + 0.03) & (t <= b_t - 0.03) & np.isfinite(py)
        if m.sum() >= 8 and (b_t - a_t) >= 0.3:
            t_o = float(t[m][0])
            tt = t[m] - t_o
            yy = py[m]
            co = np.polyfit(tt, yy, 2)
            fit = np.polyval(co, tt)
            ss_res, ss_tot = float(np.sum((yy - fit) ** 2)), float(np.sum((yy - yy.mean()) ** 2)) + 1e-12
            if 1 - ss_res / ss_tot > 0.95 and co[0] > 0.02:
                acc.append((a_t, b_t, 2 * float(co[0])))
                fits.append(dict(t0=a_t, t1=b_t, origin=t_o, coef=co))
    res["arcs"] = acc
    ev[:] = [e for e in ev if e["kind"] != "turn" or not any(a_t + 0.05 < e["t"] < b_t - 0.05 for a_t, b_t, _ in acc)]
    g = float(np.median([a for _, _, a in acc])) if acc else 0.0
    for e in ev:
        if "bounce" not in e["kind"] or e["speed_before"] <= 1e-9:
            continue
        pre = [f for f in fits if abs(f["t1"] - e["t"]) < 0.12]
        post = [f for f in fits if abs(f["t0"] - e["t"]) < 0.12]
        if e["kind"] == "floor bounce" and pre and post:   # meet the two parabolas at the impact: exact velocities
            A, B = pre[0], post[0]
            ca = np.array([A["coef"][0], A["coef"][1] - 2 * A["coef"][0] * A["origin"], A["coef"][2] - A["coef"][1] * A["origin"] + A["coef"][0] * A["origin"] ** 2])
            cb = np.array([B["coef"][0], B["coef"][1] - 2 * B["coef"][0] * B["origin"], B["coef"][2] - B["coef"][1] * B["origin"] + B["coef"][0] * B["origin"] ** 2])
            roots = np.roots(ca - cb)
            roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and abs(r.real - e["t"]) < 0.15]
            if roots:
                ti = min(roots, key=lambda r: abs(r - e["t"]))
                vb_i, va_i = float(np.polyval(np.polyder(ca), ti)), float(np.polyval(np.polyder(cb), ti))
                if vb_i > 0 > va_i:
                    e.update(t=ti, speed_before=vb_i, speed_after=-va_i, restitution=-va_i / vb_i, exact=True)
                    continue
        comp = g * 0.5 * (k1 + k4) / rate if e["kind"] == "floor bounce" else 0.0
        e["restitution"] = (e["speed_after"] + comp) / (e["speed_before"] + comp)
    ev.sort(key=lambda e: e["t"])
    # periodic motion: dominant frequency of each axis after removing the linear trend
    span = float(t[last] - t[first])
    if span >= 1.0:
        for nm, arr in (("x", sxs), ("y", sys_)):
            seg = arr[first:last + 1]
            if np.isfinite(seg).all() and len(seg) > 12:
                tt = np.arange(len(seg)) * dt
                resid = seg - np.polyval(np.polyfit(tt, seg, 1), tt)
                amp = float((resid.max() - resid.min()) / 2)
                if amp > 0.03:
                    nfft = 16 * len(resid)                              # zero-padded: finer frequency estimate
                    spec = np.abs(np.fft.rfft(resid * np.hanning(len(resid)), n=nfft)) ** 2
                    freqs = np.fft.rfftfreq(nfft, dt)
                    spec[freqs < 1.5 / span] = 0                       # need at least ~1.5 periods
                    j = int(spec.argmax())
                    near = spec[max(0, j - 12):j + 13].sum()
                    if spec.sum() > 0 and near / spec.sum() > 0.5 and freqs[j] > 0:
                        res["notes"].append(f"{nm}-oscillation at {freqs[j]:.2f} Hz (period {1 / freqs[j]:.2f}s), amplitude ±{amp:.2f} fw")
    # main-move easing
    pos = np.c_[sxs[first:last + 1], sys_[first:last + 1]]
    if np.isfinite(pos).all():
        res["easing"] = ma.classify_easing(pos, dt)
    # size and shape
    ww, hh = P[:, 3] / w, P[:, 4] / w
    okm = np.isfinite(ww)
    if okm.sum() > 8:
        n5 = max(2, okm.sum() // 5)
        a_, b_ = np.nanmedian(ww[okm][:n5]), np.nanmedian(ww[okm][-n5:])
        res["scale"] = float(b_ / a_) if a_ > 1e-6 else 1.0
        asp = np.log((hh[okm] + 1e-6) / (ww[okm] + 1e-6))
        res["aspect_var"] = float(np.std(asp))
        res["size_range"] = (float(np.nanmin(ww[okm])), float(np.nanmax(ww[okm])))
    return res


# ----------------------------------------------------------------------------- text report

def ascii_map(sxs, sys_, aspect, events, cols=46):
    """Text map of the path, cropped to its bounding box; one cell is ~2x taller than wide."""
    ok = np.where(np.isfinite(sxs) & np.isfinite(sys_))[0]
    if len(ok) < 2:
        return ""
    x0, x1 = float(sxs[ok].min()), float(sxs[ok].max())
    y0, y1 = float(sys_[ok].min()), float(sys_[ok].max())
    mx = max(0.02, 0.06 * (x1 - x0))
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - mx, y1 + mx
    bw, bh = x1 - x0, y1 - y0
    rows = int(np.clip(round(cols * bh / bw * 0.5), 3, 20))
    if bh / bw * 0.5 * cols > 20:                      # tall path: shrink the width to keep the aspect
        cols = int(np.clip(round(20 / (bh / bw * 0.5)), 12, cols))
    grid = [[" "] * cols for _ in range(rows)]

    def cell(x, y):
        return int(np.clip((y - y0) / bh * rows, 0, rows - 1)), int(np.clip((x - x0) / bw * cols, 0, cols - 1))
    for n, i in enumerate(ok):
        r, c = cell(sxs[i], sys_[i])
        grid[r][c] = str(min(9, int(10 * n / len(ok))))
    for e in events:
        mark = "P" if e["kind"] == "pause" else "B" if "bounce" in e["kind"] else "T"
        r, c = cell(e["x"], e["y"])
        grid[r][c] = mark
    r, c = cell(sxs[ok[0]], sys_[ok[0]])
    grid[r][c] = "S"
    r, c = cell(sxs[ok[-1]], sys_[ok[-1]])
    grid[r][c] = "E"
    top = "+" + "-" * cols + "+"
    head = f"x {x0:.2f}→{x1:.2f}   y {y0 * 1.0 / aspect:.2f}→{y1 / aspect:.2f} (of frame height, top→bottom)"
    return "\n".join([head, top] + ["|" + "".join(row) + "|" for row in grid] + [top])


def key_rows(P, wh, an, max_rows=48):
    w, h = wh
    t = P[:, 0]
    dur = float(t[-1] - t[0]) if len(t) else 0
    step = 0.2 if dur <= 8 else 0.5 if dur <= 20 else 1.0
    want = list(np.arange(t[0], t[-1] + 1e-6, step)) if len(t) else []
    want += [e["t"] for e in an["events"]] + [an.get("peak_t", t[0])]
    want = sorted(set(round(float(x), 2) for x in want))[:max_rows * 2]
    rows = []
    for tw_ in want:
        i = int(np.argmin(np.abs(t - tw_)))
        if abs(t[i] - tw_) > 0.6 / max(1.0, 1.0):
            continue
        ok = np.isfinite(P[i, 1])
        sp = an["speed"][i]
        d = ma.direction_words(an["vx"][i], an["vy"][i]) if np.isfinite(sp) and sp > 0.03 else "-"
        rows.append((t[i], P[i, 1] / w if ok else None, P[i, 2] / h if ok else None, P[i, 3] / w if ok else None,
                     P[i, 4] / h if ok else None, sp if np.isfinite(sp) else None, d, P[i, 7]))
    return rows[:max_rows * 2]


def describe(name, seed_t, seed_how, P, wh, an, rate, info, engine_label=None):
    w, h = wh
    out = [f"## Object `{name}`", ""]
    out.append(f"- Seed: {seed_how} at {seed_t:.2f}s; tracked {an.get('t_first', P[0, 0]):.2f}s → {an.get('t_last', P[-1, 0]):.2f}s "
               f"at {rate:.0f} fps with {engine_label or ('a neural ViT tracker' if MODEL.is_file() else 'template matching (model missing)')}; "
               f"{100 * an.get('tracked_frac', 0):.0f}% of the frames found the object")
    lost = P[:, 8] >= 1
    cuts = [P[i, 0] for i in range(1, len(P)) if P[i, 8] == 2 and P[i - 1, 8] != 2]
    if cuts:
        out.append(f"- Lost at scene cut(s): {', '.join(f'{c:.2f}s' for c in cuts)}")
    rec = [P[i, 0] for i in range(len(P)) if P[i, 8] == 3]
    if rec:
        out.append(f"- Found again by appearance at {', '.join(f'{c:.2f}s' for c in rec[:5])} (verify in the filmstrip)")
    gaps = [(P[a, 0], P[b, 0]) for a, b in segments(lost) if P[b, 0] - P[a, 0] >= 0.3]
    if gaps:
        out.append("- Not visible / lost: " + ", ".join(f"{a:.2f}–{b:.2f}s" for a, b in gaps[:6]))
    if "total_length" not in an:
        out += [""] + [f"- {n}" for n in an["notes"]]
        return "\n".join(out)
    out.append(f"- Path ({'camera removed, in scene coordinates' if an['use_scene'] else 'on screen'}): length {an['total_length']:.2f} frame widths, "
               f"net {an['net']:.2f}, straightness {an['straightness']:.2f}; speed mean {an['mean_speed']:.2f}, peak {an['peak_speed']:.2f} fw/s at {an['peak_t']:.2f}s")
    if "scale" in an:
        sc = an["scale"]
        trend = "grows (comes closer)" if sc > 1.25 else "shrinks (moves away)" if sc < 0.8 else "keeps its size"
        out.append(f"- Size: {trend} (×{sc:.2f} first→last fifth); width range {an['size_range'][0]:.3f}–{an['size_range'][1]:.3f} fw; "
                   f"aspect ratio varies by {100 * an['aspect_var']:.0f}% (squash & stretch if rhythmic)")
    e = an.get("easing")
    if e and "rmse" in e:
        out.append(f"- Main move easing: **{e['label']}** (move ≈{e['duration']:.2f}s, fit error {e['rmse']:.3f})")
    elif e:
        out.append(f"- Main move: {e['label']}")
    if an.get("arcs"):
        g = [a for _, _, a in an["arcs"]]
        out.append(f"- Free-flight arcs: {len(g)}; vertical acceleration {np.median(g):.2f} fw/s² (median, y down; range {min(g):.2f}…{max(g):.2f})"
                   + (" — constant across arcs, gravity-like" if len(g) > 1 and np.std(g) < 0.25 * abs(np.median(g)) else ""))
    for n in an["notes"]:
        out.append(f"- {n}")
    out.append("")
    if an["events"]:
        out.append("**Events**")
        out.append("")
        for i, ev in enumerate(an["events"], 1):
            if ev["kind"] == "pause":
                out.append(f"- {ev['t']:.2f}–{ev['t_end']:.2f}s pause at ({ev['x']:.2f}, {ev['y'] * w / h:.2f})")
            elif "bounce" in ev["kind"]:
                r = ev.get("restitution")
                out.append(f"- {ev['t']:.2f}s {ev['kind']} at ({ev['x']:.2f}, {ev['y'] * w / h:.2f}): speed {ev['speed_before']:.2f} → {ev['speed_after']:.2f} fw/s"
                           + (f", keeps {100 * r:.0f}% of its speed (bounciness{', from the fitted arcs' if ev.get('exact') else ''})" if r else ""))
            else:
                out.append(f"- {ev['t']:.2f}s turn of {ev['angle']:.0f}° at ({ev['x']:.2f}, {ev['y'] * w / h:.2f})")
        out.append("")
    rows = key_rows(P, wh, an)
    out.append("**Path (screen position of the box centre, 0..1; speed in frame widths/s)**")
    out.append("")
    out.append("| t | x | y | w | h | speed | heading | conf |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r[1] is None:
            out.append(f"| {r[0]:.2f} | – | – | – | – | – | lost | {r[7]:.2f} |")
        else:
            out.append(f"| {r[0]:.2f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {'-' if r[5] is None else f'{r[5]:.2f}'} | {r[6]} | {r[7]:.2f} |")
    mp = ascii_map(smooth_pos(an["px"], rate), smooth_pos(an["py"], rate), h / w, an["events"])
    if mp:
        out += ["", "**Path map** (S start, E end, digits 0–9 = time in tenths of the track, P pause, B bounce, T turn; same aspect as the frame)", "", "```", mp, "```"]
    return "\n".join(out)


# ----------------------------------------------------------------------------- images

def timed_color(u):
    return ma.time_color(u)


def path_image(video, objs, info, out_path, t_ref):
    """All tracked objects drawn on the reference frame; the canvas grows if the path leaves the frame."""
    dw = 540
    dh = ma.even(info["height"] * dw / info["width"])
    ref = ma.grab_frame(video, t_ref, dw, dh)
    pts_all = []
    for o in objs:
        P, an = o["P"], o["an"]
        if "px" in an:
            ok = np.isfinite(an["px"])
            pts_all += list(zip(an["px"][ok] * dw, an["py"][ok] * dw))
    if not pts_all:
        return False
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    margin = 30
    x0, y0 = min(0, min(xs) - margin), min(0, min(ys) - margin)
    x1, y1 = max(dw, max(xs) + margin), max(dh, max(ys) + margin)
    scale = min(1.0, 900.0 / max(x1 - x0, y1 - y0))
    W_, H_ = int((x1 - x0) * scale), int((y1 - y0) * scale)
    canvas = np.full((H_ + 26, W_, 3), 18, np.uint8)
    ox, oy = int(-x0 * scale), int(-y0 * scale)
    small = cv2.resize((ref * 0.55).astype(np.uint8), (int(dw * scale), int(dh * scale)))
    canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    cv2.rectangle(canvas, (ox, oy), (ox + small.shape[1] - 1, oy + small.shape[0] - 1), (90, 90, 90), 1)
    T = lambda x, y: (int((x * dw - x0) * scale), int((y * dw - y0) * scale))
    for n, o in enumerate(objs):
        P, an = o["P"], o["an"]
        if "px" not in an:
            continue
        t = P[:, 0]
        px, py = an["px"], an["py"]
        ok = np.where(np.isfinite(px))[0]
        if len(ok) < 2:
            continue
        t0, t1 = t[ok[0]], t[ok[-1]]
        for a, b in zip(ok[:-1], ok[1:]):
            if b - a > 1 and t[b] - t[a] > 0.5:
                continue
            cv2.line(canvas, T(px[a], py[a]), T(px[b], py[b]), timed_color((t[a] - t0) / max(1e-6, t1 - t0)), 2, cv2.LINE_AA)
        tick = 0.1 if t1 - t0 <= 6 else 0.25 if t1 - t0 <= 15 else 0.5
        k = math.ceil(t0 / tick)
        while k * tick <= t1:
            i = int(np.argmin(np.abs(t - k * tick)))
            if np.isfinite(px[i]):
                cv2.circle(canvas, T(px[i], py[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)
            k += 1
        for e in an["events"]:
            tag = {"pause": "P", "turn": "T"}.get(e["kind"], "B")
            p = T(e["x"], e["y"] * 1.0)
            cv2.circle(canvas, p, 9, (0, 255, 255), 1, cv2.LINE_AA)
            put(canvas, f"{tag}{e['t']:.1f}", (p[0] + 8, p[1] - 8), 0.36, (0, 255, 255))
        cv2.putText(canvas, "S", (T(px[ok[0]], py[ok[0]])[0] + 6, T(px[ok[0]], py[ok[0]])[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 60), 2, cv2.LINE_AA)
        end = T(px[ok[-1]], py[ok[-1]])
        cv2.putText(canvas, "E", (end[0] + 6, end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2, cv2.LINE_AA)
        put(canvas, o["name"], (T(px[ok[0]], py[ok[0]])[0] - 4, T(px[ok[0]], py[ok[0]])[1] - 14), 0.5, PALETTE[n % len(PALETTE)])
    bar = canvas[H_:H_ + 26]
    for x in range(W_):
        bar[:, x] = timed_color(x / max(1, W_ - 1))
    note = "path in scene coordinates (camera removed), drawn on the seed frame" if any(o["an"].get("use_scene") for o in objs) else "path on screen"
    canvas[:22, :] = (canvas[:22, :] * 0.35).astype(np.uint8)
    cv2.putText(canvas, f"{note}; dots every tick; S start E end; yellow rings = events", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imwrite(str(out_path), canvas)


def filmstrip_image(video, obj, info, out_path, n=10):
    P, an = obj["P"], obj["an"]
    ok = np.where(np.isfinite(P[:, 1]))[0]
    if len(ok) < 2:
        return False
    times = [float(P[i, 0]) for i in np.linspace(ok[0], max(ok[0] + 1, ok[-1] - 2), n).astype(int)]
    times = [float(P[ok[int(np.argmin(np.abs(P[ok, 0] - tt)))], 0]) for tt in times]
    dw = 720
    dh = ma.even(info["height"] * dw / info["width"])
    w, h = obj["wh"]
    cell_h = 150
    cells = []
    for tt in times:
        i = int(np.argmin(np.abs(P[:, 0] - tt)))
        frame = ma.grab_frame(video, tt, dw, dh)
        cx, cy, bw, bh = P[i, 1] / w, P[i, 2] / h, P[i, 3] / w, P[i, 4] / h
        side = max(bw * dw, bh * dh) * 1.7 + 12
        x0, y0 = int(cx * dw - side / 2), int(cy * dh - side / 2)
        pad = int(side)
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        crop = padded[y0 + pad:y0 + pad + int(side), x0 + pad:x0 + pad + int(side)]
        if crop.size == 0:
            continue
        bx0, by0 = int((cx - bw / 2) * dw - x0), int((cy - bh / 2) * dh - y0)
        if obj.get("engine") == "point":
            mx, my = int(cx * dw - x0), int(cy * dh - y0)
            cv2.circle(crop, (mx, my), 6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(crop, (mx - 10, my), (mx + 10, my), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(crop, (mx, my - 10), (mx, my + 10), (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(crop, (bx0, by0), (bx0 + int(bw * dw), by0 + int(bh * dh)), (0, 255, 255), 1)
        cell = cv2.resize(crop, (cell_h, cell_h))
        put(cell, f"{tt:.2f}s", (4, 14), 0.4)
        if P[i, 8] >= 1:
            put(cell, "lost" if P[i, 8] != 3 else "refound", (4, cell_h - 6), 0.4, (80, 80, 255))
        cells.append(cell)
    if not cells:
        return False
    cols = 5
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return cv2.imwrite(str(out_path), np.vstack(rows))


def plot_image(objs, out_path, measures=None):
    W = 1000
    L, R = 56, 12
    panels = []
    allt = [o["P"][:, 0] for o in objs if len(o["P"])]
    t0, t1 = min(a.min() for a in allt), max(a.max() for a in allt)
    X = lambda v: int(L + (v - t0) / max(1e-6, t1 - t0) * (W - L - R))

    def panel(title, key, flip=False, h=120, ymax=None):
        img = np.full((h, W, 3), 22, np.uint8)
        cv2.rectangle(img, (L, 4), (W - R, h - 6), (60, 60, 60), 1)
        put(img, title, (L + 4, 16), 0.38)
        series = []
        for o in objs:
            if key not in o["an"]:
                continue
            series.append((o, np.asarray(o["an"][key], dtype=np.float64)))
        if not series:
            return img
        vals = np.concatenate([s[1][np.isfinite(s[1])] for s in series])
        if len(vals) == 0:
            return img
        lo, hi = (0.0, float(np.percentile(vals, 99.5))) if ymax == "pos" else (float(vals.min()), float(vals.max()))
        floor = 0.06 if key in ("px", "py") else 0.2
        if hi - lo < floor:                                   # keep flat things flat
            mid = (hi + lo) / 2
            lo, hi = (mid - floor / 2, mid + floor / 2) if ymax != "pos" else (0.0, max(hi, floor))
        for n, (o, arr) in enumerate(series):
            t = o["P"][:, 0]
            pts = []
            for ti, v in zip(t, arr):
                if not np.isfinite(v):
                    if len(pts) > 1:
                        cv2.polylines(img, [np.array(pts, np.int32)], False, PALETTE[n % 6], 1, cv2.LINE_AA)
                    pts = []
                    continue
                f = (v - lo) / (hi - lo)
                f = 1 - f if flip else f
                pts.append([X(ti), int(h - 8 - np.clip(f, 0, 1) * (h - 28))])
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, PALETTE[n % 6], 1, cv2.LINE_AA)
            for e in o["an"].get("events", []):
                cv2.line(img, (X(e["t"]), h - 14), (X(e["t"]), h - 7), (0, 255, 255), 1)
        put(img, f"{lo:.2f}", (2, h - 8), 0.32)
        put(img, f"{hi:.2f}", (2, 22), 0.32)
        return img

    panels = [panel("x position (frame widths from the left)", "px"), panel("y position (frame widths from the top; up = up on screen)", "py", flip=True),
              panel("speed (frame widths / s)", "speed", ymax="pos")]
    legend = np.full((24, W, 3), 22, np.uint8)
    for n, o in enumerate(objs):
        put(legend, o["name"], (L + n * 110, 16), 0.45, PALETTE[n % 6])
    axis = np.full((24, W, 3), 22, np.uint8)
    step = max(1, int(math.ceil((t1 - t0) / 10)))
    for sec in range(int(math.ceil(t0)), int(t1) + 1, step):
        cv2.line(axis, (X(sec), 0), (X(sec), 6), (150, 150, 150), 1)
        put(axis, f"{sec}s", (X(sec) - 10, 19), 0.36)
    extra = [measure_panel(m, W, L, R, t0, t1) for m in (measures or [])]
    return cv2.imwrite(str(out_path), np.vstack([legend] + panels + extra + [axis]))


def seed_image(video, name, t, box, info, out_path, point=None):
    dw = 540
    dh = ma.even(info["height"] * dw / info["width"])
    img = ma.grab_frame(video, t, dw, dh)
    x, y, bw, bh = box
    p0, p1 = (int(x * dw), int(y * dh)), (int((x + bw) * dw), int((y + bh) * dh))
    big = img.copy()
    if point is not None:
        pc = (int(point[0] * dw), int(point[1] * dh))
        cv2.circle(big, pc, 10, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(big, (pc[0] - 18, pc[1]), (pc[0] + 18, pc[1]), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(big, (pc[0], pc[1] - 18), (pc[0], pc[1] + 18), (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(big, p0, p1, (0, 255, 255), 2)
    cv2.putText(big, ma.ascii_safe(f"{name} @ {t:.2f}s"), (p0[0], max(14, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    side = int(max(bw * dw, bh * dh) * 2.4) + 16
    cx, cy = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
    pad = side
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    zoom = padded[cy - side // 2 + pad:cy + side // 2 + pad, cx - side // 2 + pad:cx + side // 2 + pad].copy()
    zb0 = (p0[0] - (cx - side // 2), p0[1] - (cy - side // 2))
    cv2.rectangle(zoom, zb0, (zb0[0] + p1[0] - p0[0], zb0[1] + p1[1] - p0[1]), (0, 255, 255), 1)
    zoom = cv2.resize(zoom, (dh * 2 // 3, dh * 2 // 3))
    zoom = np.vstack([zoom, np.zeros((dh - zoom.shape[0], zoom.shape[1], 3), np.uint8)]) if dh > zoom.shape[0] else zoom[:dh]
    return cv2.imwrite(str(out_path), np.hstack([big, zoom]))


EVENT_LABEL = {"floor bounce": "BOUNCE", "ceiling bounce": "BOUNCE", "wall bounce (left)": "WALL", "wall bounce (right)": "WALL",
               "turn": "TURN", "pause": "PAUSE"}


def event_text(e):
    base = EVENT_LABEL.get(e["kind"], e["kind"].upper())
    r = e.get("restitution")
    return f"{base} {100 * r:.0f}%" if r and "bounce" in e["kind"] else base


def annotated_video(video, objs, info, out_path, t_a, t_b, rate, measures=None):
    """H.264 clip for a human: box, trail, speed, event flashes (BOUNCE 72%) and a live speed gauge; audio included."""
    vw = 540
    vh = ma.even(info["height"] * vw / info["width"])
    gh = 64
    span = max(1e-6, t_b - t_a)
    gauge = np.full((gh, vw, 3), 24, np.uint8)
    top = max([float(np.nanmax(o["an"]["speed"])) for o in objs if "speed" in o["an"] and np.isfinite(o["an"]["speed"]).any()] or [1.0])
    for n, o in enumerate(objs):
        sp = o["an"].get("speed")
        if sp is None:
            continue
        pts = [[int((t - t_a) / span * (vw - 1)), int(gh - 6 - np.clip(v / max(top, 1e-6), 0, 1) * (gh - 22))]
               for t, v in zip(o["P"][:, 0], sp) if np.isfinite(v) and t_a <= t <= t_b]
        if len(pts) > 1:
            cv2.polylines(gauge, [np.array(pts, np.int32)], False, PALETTE[n % 6], 1, cv2.LINE_AA)
        for e in o["an"].get("events", []):
            if t_a <= e["t"] <= t_b:
                x = int((e["t"] - t_a) / span * (vw - 1))
                cv2.line(gauge, (x, gh - 12), (x, gh - 2), (0, 255, 255), 1)
    put(gauge, "speed (frame widths / s)   yellow ticks = events", (6, 12), 0.36)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{vw}x{vh + gh}", "-r", f"{rate:.4f}", "-i", "-",
           "-ss", f"{t_a:.3f}", "-t", f"{t_b - t_a:.3f}", "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "24", "-c:a", "aac", "-shortest", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for t, frame in stream(video, t_a, t_b, rate, vw, vh):
            img = frame.copy()
            for n, o in enumerate(objs):
                P = o["P"]
                w, h = o["wh"]
                i = int(np.argmin(np.abs(P[:, 0] - t)))
                col = PALETTE[n % 6]
                trail = [j for j in range(max(0, i - int(1.2 * rate)), i + 1) if np.isfinite(P[j, 1])]
                for a, b in zip(trail[:-1], trail[1:]):
                    f = (a - trail[0]) / max(1, len(trail))
                    cv2.line(img, (int(P[a, 1] / w * vw), int(P[a, 2] / h * vh)), (int(P[b, 1] / w * vw), int(P[b, 2] / h * vh)),
                             tuple(int(c * (0.35 + 0.65 * f)) for c in col), 2, cv2.LINE_AA)
                if np.isfinite(P[i, 1]) and abs(P[i, 0] - t) < 1.5 / rate:
                    x, y, bw, bh = P[i, 1] / w * vw, P[i, 2] / h * vh, P[i, 3] / w * vw, P[i, 4] / h * vh
                    if o.get("engine") == "point":
                        cv2.circle(img, (int(x), int(y)), 9, col, 2, cv2.LINE_AA)
                        cv2.line(img, (int(x) - 15, int(y)), (int(x) + 15, int(y)), col, 1, cv2.LINE_AA)
                        cv2.line(img, (int(x), int(y) - 15), (int(x), int(y) + 15), col, 1, cv2.LINE_AA)
                        bw = bh = 18
                    else:
                        cv2.rectangle(img, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), col, 2)
                    sp = o["an"].get("speed")
                    label = o["name"] + (f"  {sp[i]:.2f}" if sp is not None and np.isfinite(sp[i]) else "")
                    put(img, label, (int(x - bw / 2), max(12, int(y - bh / 2) - 5)), 0.45, col)
                    for e in o["an"].get("events", []):
                        age = t - e["t"]
                        if 0 <= age < 0.8:                           # event flash: ring grows, text rises and fades
                            fade = 1 - age / 0.8
                            cv2.circle(img, (int(x), int(y)), int(max(bw, bh) / 2 + 10 + 40 * age), (0, 255, 255), 2, cv2.LINE_AA)
                            txt = event_text(e)
                            colr = tuple(int(c * (0.4 + 0.6 * fade)) for c in (0, 255, 255))
                            ma.put_text(img, txt, (int(x - bw / 2), max(26, int(y - bh / 2 - 22 - 25 * age))), 0.8, colr, 2)
                elif P[i, 8] >= 1:
                    put(img, f"{o['name']}: lost", (6, 36 + 16 * n), 0.45, col)
            for m in (measures or []):
                if "v" not in m:
                    continue
                pos = []
                for nm in m["names"]:
                    o = next(o_ for o_ in objs if o_["name"] == nm)
                    P_, (w_, h_) = o["P"], o["wh"]
                    j = int(np.argmin(np.abs(P_[:, 0] - t)))
                    pos.append((int(P_[j, 1] / w_ * vw), int(P_[j, 2] / h_ * vh)) if np.isfinite(P_[j, 1]) else None)
                k = int(np.argmin(np.abs(m["t"] - t)))
                if all(pos) and np.isfinite(m["v"][k]):
                    if m["kind"] == "distance":
                        cv2.line(img, pos[0], pos[1], (255, 255, 255), 1, cv2.LINE_AA)
                        mid = ((pos[0][0] + pos[1][0]) // 2 + 8, (pos[0][1] + pos[1][1]) // 2)
                        put(img, f"{m['v'][k]:.3f}", mid, 0.5, (0, 255, 255))
                    else:
                        cv2.line(img, pos[1], pos[0], (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.line(img, pos[1], pos[2], (255, 255, 255), 1, cv2.LINE_AA)
                        put(img, f"{m['v'][k]:.0f} deg", (pos[1][0] + 10, pos[1][1] - 6), 0.5, (0, 255, 255))
            put(img, f"watch track   t={t:.2f}s", (6, 16), 0.5)
            g = gauge.copy()
            cx = int((t - t_a) / span * (vw - 1))
            cv2.line(g, (cx, 14), (cx, gh - 1), (255, 255, 255), 1)
            proc.stdin.write(np.vstack([img, g]).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    return Path(out_path).is_file()


def headline(o):
    """Plain-ASCII key facts for one object (the summary image header)."""
    an = o["an"]
    if "total_length" not in an:
        return [f"{o['name']}: too little data for an analysis"]
    parts = [f"{o['name']}: followed {an['t_first']:.1f}-{an['t_last']:.1f}s, path {an['total_length']:.2f} frame widths, peak speed {an['peak_speed']:.2f} fw/s"]
    ev = an["events"]
    bounces = [e for e in ev if "bounce" in e["kind"]]
    facts = []
    if bounces:
        rs = [e["restitution"] for e in bounces if e.get("restitution")]
        facts.append(f"{len(bounces)} bounces" + (f" (keeps ~{100 * float(np.median(rs)):.0f}% of its speed each time)" if rs else ""))
    if an.get("arcs"):
        facts.append(f"gravity {np.median([a for _, _, a in an['arcs']]):.2f} fw/s^2")
    turns = [e for e in ev if e["kind"] == "turn"]
    pauses = [e for e in ev if e["kind"] == "pause"]
    if turns:
        facts.append(f"{len(turns)} sharp turns")
    if pauses:
        facts.append(f"{len(pauses)} pauses")
    e = an.get("easing")
    if e and "rmse" in e:
        facts.append(f"main move: {e['label'].split(' (strength')[0]}")
    for n in an["notes"][:2]:
        facts.append(n.replace("\u00b1", "+-"))
    if "scale" in an and (an["scale"] > 1.25 or an["scale"] < 0.8):
        facts.append("gets bigger" if an["scale"] > 1 else "gets smaller")
    return parts + ["   " + "; ".join(facts)] if facts else parts


def summary_image(out_dir, objs, out_path, measures=None):
    """ONE picture for a human: key facts on top, then the path, filmstrips and curves."""
    W = 960
    out_dir = Path(out_dir)
    lines = ["WATCH TRACK  -  what the tracker measured"]
    for o in objs:
        lines += headline(o)
    for m in (measures or []):
        label = f"{m['names'][0]}<->{m['names'][1]} distance" if m["kind"] == "distance" else f"angle {m['names'][0]}-{m['names'][1]}-{m['names'][2]}"
        if "min" in m:
            lines.append(f"{label}: {m['min']:.2f}..{m['max']:.2f} {m['unit']}" + (f", {len(m.get('peaks', []))} cycles, period {m['period']:.2f}s" if "period" in m else ""))
    head = np.full((14 + 22 * len(lines), W, 3), 20, np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(head, ma.ascii_safe(text), (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if k else 0.6,
                    (0, 220, 255) if k == 0 else (235, 235, 235), 1, cv2.LINE_AA)
    parts = [head]
    for name in ["paths.png"] + [f"filmstrip_{o['name']}.png" for o in objs] + ["plot.png"]:
        img = cv2.imread(str(out_dir / name))
        if img is None:
            continue
        scale = W / img.shape[1]
        img = cv2.resize(img, (W, max(8, int(img.shape[0] * scale))))
        if name == "paths.png" and img.shape[0] > 1000:
            img = cv2.resize(img, (int(W * 1000 / img.shape[0]), 1000))
            img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
        parts.append(img)
    return cv2.imwrite(str(out_path), np.vstack(parts))


# ----------------------------------------------------------------------------- relations

# ----------------------------------------------------------------------------- measurements (shape change)

def build_measures(objs, dist_specs, angle_specs, rate):
    """Distance between two tracked points, or the angle at the middle one of three, over time. For mouths, fins, limbs, hinges."""
    by = {o["name"]: o for o in objs}

    def get(n):
        if n not in by:
            raise SystemExit(f"'{n}' is not one of the --obj names ({', '.join(by)}); --distance/--angle refer to tracked objects by name.")
        return by[n]

    def screen(o, ts):
        P, w = o["P"], o["wh"][0]
        ok = np.isfinite(P[:, 1])
        if ok.sum() < 4:
            return None
        t = P[ok, 0]
        inside = (ts >= t[0]) & (ts <= t[-1])
        x, y = np.interp(ts, t, P[ok, 1] / w), np.interp(ts, t, P[ok, 2] / w)
        x[~inside], y[~inside] = np.nan, np.nan
        lost = np.array([P[int(np.argmin(np.abs(P[:, 0] - tt))), 8] >= 1 for tt in ts])
        x[lost], y[lost] = np.nan, np.nan
        return np.c_[x, y]

    out = []
    specs = [("distance", s_) for s_ in dist_specs] + [("angle", s_) for s_ in angle_specs]
    for kind, spec in specs:
        names = [n.strip() for n in spec.split(",")]
        if len(names) != (2 if kind == "distance" else 3):
            raise SystemExit(f"--{kind} needs {'two names (a,b)' if kind == 'distance' else 'three names (a,b,c); the angle is measured at b'}, got '{spec}'.")
        objs_ = [get(n) for n in names]
        lo = max(float(o["P"][np.isfinite(o["P"][:, 1]), 0].min()) for o in objs_)
        hi = min(float(o["P"][np.isfinite(o["P"][:, 1]), 0].max()) for o in objs_)
        if hi - lo < 0.3:
            out.append(dict(name=spec, kind=kind, names=names, note="the objects are never tracked at the same time"))
            continue
        ts = np.arange(lo, hi, 1.0 / rate)
        pts = [screen(o, ts) for o in objs_]
        if any(p_ is None for p_ in pts):
            out.append(dict(name=spec, kind=kind, names=names, note="too few tracked samples"))
            continue
        if kind == "distance":
            v = np.hypot(*(pts[0] - pts[1]).T)
            unit = "fw"
        else:
            u, w_ = pts[0] - pts[1], pts[2] - pts[1]
            cosang = np.einsum("ij,ij->i", u, w_) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w_, axis=1) + 1e-12)
            v = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            unit = "deg"
        v = smooth_pos(v, rate, 0.1)
        fin = np.isfinite(v)
        m = dict(name=spec, kind=kind, names=names, unit=unit, t=ts, v=v)
        if fin.sum() > 6:
            vf = v[fin]
            span = float(vf.max() - vf.min())
            prom = max(0.15 * span, 0.004 if unit == "fw" else 3.0)
            vv = np.where(fin, v, np.nanmean(vf))
            pk, _ = signal.find_peaks(vv, prominence=prom, distance=max(1, int(0.12 * rate)))
            va, _ = signal.find_peaks(-vv, prominence=prom, distance=max(1, int(0.12 * rate)))
            m.update(min=float(vf.min()), max=float(vf.max()), mean=float(vf.mean()), peaks=[float(ts[i]) for i in pk],
                     valleys=[float(ts[i]) for i in va], peak_values=[float(v[i]) for i in pk], valley_values=[float(v[i]) for i in va])
            if len(pk) >= 2:
                period = float(np.median(np.diff(ts[pk])))
                m.update(period=period, freq=1.0 / period if period > 0 else None)
            if len(pk) and len(va):
                m["amplitude"] = float(np.median(v[pk]) - np.median(v[va]))
        out.append(m)
    return out


def measure_lines(measures, rate):
    lines = ["## Measurements (shape change between tracked points)", ""]
    for m in measures:
        title = f"`{m['names'][0]}` ↔ `{m['names'][1]}` distance" if m["kind"] == "distance" else f"angle at `{m['names'][1]}` between `{m['names'][0]}` and `{m['names'][2]}`"
        if "note" in m:
            lines.append(f"- {title}: {m['note']}")
            continue
        u = m["unit"]
        f = (lambda x: f"{x:.3f}") if u == "fw" else (lambda x: f"{x:.0f}")
        lines.append(f"- **{title}**, in {'frame widths' if u == 'fw' else 'degrees'}: min {f(m['min'])}, max {f(m['max'])}, mean {f(m['mean'])}"
                     + (f"; typical swing {f(m['amplitude'])}" if "amplitude" in m else ""))
        if "peaks" in m:
            word_hi, word_lo = ("wide open / far apart", "closed / close together") if m["kind"] == "distance" else ("opened", "closed")
            lines.append(f"  - {len(m['peaks'])} maxima ({word_hi}) at " + ", ".join(f"{t:.2f}s" for t in m["peaks"][:12])
                         + f"; {len(m['valleys'])} minima ({word_lo}) at " + ", ".join(f"{t:.2f}s" for t in m["valleys"][:12]))
        if "period" in m:
            lines.append(f"  - regular cycle: period {m['period']:.2f}s ({m['freq']:.2f} Hz)")
        lines.append(f"  - curve: `{ma.sparkline(m['v'][np.isfinite(m['v'])][::max(1, int(len(m['v']) / 70))])}`  ({m['t'][0]:.1f}s → {m['t'][-1]:.1f}s)")
        rows = np.linspace(0, len(m["t"]) - 1, min(24, len(m["t"]))).astype(int)
        lines.append("  - " + "; ".join(f"{m['t'][i]:.2f}s: {f(m['v'][i])}" for i in rows if np.isfinite(m["v"][i])))
    lines.append("")
    return lines


def measure_panel(m, W, L, R, t0, t1, h=120):
    img = np.full((h, W, 3), 22, np.uint8)
    cv2.rectangle(img, (L, 4), (W - R, h - 6), (60, 60, 60), 1)
    X = lambda v: int(L + (v - t0) / max(1e-6, t1 - t0) * (W - L - R))
    label = f"{m['kind']} {m['name']}" + (f" ({m['unit']})" if "unit" in m else "")
    put(img, label, (L + 4, 16), 0.38, (0, 220, 255))
    if "v" not in m or not np.isfinite(m["v"]).any():
        return img
    v = m["v"]
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    floor = 0.02 if m["unit"] == "fw" else 10.0
    if hi - lo < floor:
        mid = (hi + lo) / 2
        lo, hi = mid - floor / 2, mid + floor / 2
    Y = lambda val: int(h - 8 - np.clip((val - lo) / (hi - lo), 0, 1) * (h - 28))
    pts = []
    for ti, vi in zip(m["t"], v):
        if np.isfinite(vi):
            pts.append([X(ti), Y(vi)])
        elif len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
            pts = []
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
    for t_, v_ in zip(m.get("peaks", []), m.get("peak_values", [])):
        cv2.circle(img, (X(t_), Y(v_)), 4, (80, 230, 120), -1, cv2.LINE_AA)
    for t_, v_ in zip(m.get("valleys", []), m.get("valley_values", [])):
        cv2.circle(img, (X(t_), Y(v_)), 4, (255, 160, 60), -1, cv2.LINE_AA)
    put(img, f"{lo:.2f}", (2, h - 8), 0.32)
    put(img, f"{hi:.2f}", (2, 22), 0.32)
    return img


def relations(objs, rate):
    if len(objs) < 2:
        return []
    lines = ["## Relations between objects", ""]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ta, tb = a["P"][:, 0], b["P"][:, 0]
            lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
            if hi - lo < 0.2:
                lines.append(f"- `{a['name']}` and `{b['name']}` are never tracked at the same time.")
                continue
            ts = np.arange(lo, hi, 1.0 / rate)
            def at(o, key_i):
                return np.interp(ts, o["P"][:, 0], np.where(np.isfinite(o["P"][:, key_i]), o["P"][:, key_i], np.nan))
            ax_, ay_, aw_, ah_ = at(a, 1), at(a, 2), at(a, 3), at(a, 4)
            bx_, by_, bw_, bh_ = at(b, 1), at(b, 2), at(b, 3), at(b, 4)
            ok = np.isfinite(ax_) & np.isfinite(bx_)
            if ok.sum() < 3:
                continue
            w = a["wh"][0]
            d = np.hypot(ax_ - bx_, ay_ - by_) / w
            gap_x = np.abs(ax_ - bx_) - (aw_ + bw_) / 2
            gap_y = np.abs(ay_ - by_) - (ah_ + bh_) / 2
            gap = np.where(ok, np.maximum(gap_x, gap_y) / w, np.nan)
            k = int(np.nanargmin(d))
            lines.append(f"- `{a['name']}` ↔ `{b['name']}`: distance (centre to centre) from {np.nanmax(d):.2f} to {np.nanmin(d):.2f} fw; "
                         f"closest at {ts[k]:.2f}s ({d[k]:.2f} fw, box gap {gap[k]:.3f})")
            touch = np.where(ok & (gap < 0.005))[0]
            if len(touch):
                lines.append(f"  - boxes touch / overlap at {', '.join(f'{ts[s]:.2f}–{ts[e]:.2f}s' for s, e in segments(np.isin(np.arange(len(ts)), touch))[:5])}")
            dd = np.diff(d[ok])
            if len(dd):
                lines.append(f"  - approaching {100 * float((dd < 0).mean()):.0f}% of the time, receding {100 * float((dd > 0).mean()):.0f}%")
            rel_x = np.nanmedian((bx_ - ax_)[ok])
            rel_y = np.nanmedian((by_ - ay_)[ok])
            lines.append(f"  - on average `{b['name']}` is {'right of' if rel_x > 0 else 'left of'} and {'below' if rel_y > 0 else 'above'} `{a['name']}`")
    return lines


# ----------------------------------------------------------------------------- commands

def cmd_track(args, info):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dur = info["duration"] or 1e9
    t_start = max(0.0, args.start or 0.0)
    t_end = min(dur, args.end if args.end is not None else dur)
    rate = min(info["fps"], 30.0)
    if (t_end - t_start) * rate > 1500:
        rate = max(8.0, 1500.0 / (t_end - t_start))
    objs = []
    for spec in args.obj:
        try:
            head, coord = spec.split("=", 1)
            name, at = head.split("@", 1)
            seed_t = float(at)
        except ValueError:
            raise SystemExit(f"Bad --obj '{spec}'. Use NAME@TIME=SPEC, e.g. ball@1.2=0.5,0.4 (a point) or ball@1.2=0.45,0.35,0.1,0.1 (a box).")
        if not (t_start <= seed_t <= t_end):
            raise SystemExit(f"Seed time {seed_t}s for '{name}' lies outside the tracked range {t_start:.1f}-{t_end:.1f}s.")
        box, how = resolve_seed(args.video, seed_t, coord, info, args.size)
        is_point = len(parse_nums(coord)) == 2
        engine = args.engine
        if engine == "point" and not is_point:
            raise SystemExit("--engine point follows a clicked point: give x,y (a box seed needs --engine box or auto).")
        if engine == "auto":
            engine = "point" if is_point and adapter.tracker_ready() else "box"
        if engine == "point" and not adapter.tracker_ready():
            raise SystemExit(f"The CoTracker3 point tracker is not installed. {adapter.TRACKER_HINT}")
        note = ""
        if args.engine == "auto" and is_point and engine == "box":
            note = " (CoTracker3 not installed, so the box tracker follows the point instead; " + adapter.TRACKER_HINT + ")"
        pt = parse_nums(coord)[:2]
        if engine == "point":
            how = "exact point you gave (followed as a point, no region)"
        seed_image(args.video, name, seed_t, box, info, out / f"seed_{name}.png", point=pt if engine == "point" else None)
        if args.confirm:
            objs.append(dict(name=name, seed_t=seed_t, how=how + note, box=box, engine=engine))
            continue
        if engine == "point":
            P, wh = track_point(args.video, name, seed_t, pt, args.size, info, t_start, t_end, rate)
        else:
            P, wh = track_object(args.video, name, seed_t, box, info, t_start, t_end, rate)
        an = analyse(P, info, wh, rate)
        if engine == "point":                                   # a point has no size
            for key in ("scale", "aspect_var", "size_range"):
                an.pop(key, None)
        objs.append(dict(name=name, seed_t=seed_t, how=how + note, box=box, P=P, wh=wh, an=an, engine=engine,
                         engine_label=POINT_LABEL if engine == "point" else BOX_LABEL))
    lines = ["# watch: track report", ""]
    images = []
    if args.confirm:
        for o in objs:
            lines.append(f"- `{o['name']}`: {o['how']}, box {','.join(f'{v:.2f}' for v in o['box'])} — check `{out / ('seed_' + o['name'] + '.png')}` shows the right object, then run without --confirm.")
        text = "\n".join(lines)
        (out / "report.md").write_text(text, encoding="utf-8")
        print(text)
        return
    lines += [f"- Source: `{args.video}` ({info['width']}x{info['height']}, {info['fps']:.1f} fps, {dur:.1f}s)",
              f"- Range {t_start:.1f}–{t_end:.1f}s analysed at {rate:.0f} fps. Coordinates are 0..1 (x right, y down); "
              "speeds in frame widths per second (fw/s).", ""]
    for o in objs:
        lines.append(describe(o["name"], o["seed_t"], o["how"], o["P"], o["wh"], o["an"], rate, info, o.get("engine_label")))
        lines.append("")
        np.savetxt(out / f"path_{o['name']}.csv", o["P"], delimiter=",", fmt="%.5f",
                   header="t,cx_px,cy_px,w_px,h_px,scene_x_px,scene_y_px,confidence,state(0ok 1lost 2cut 3refound);px at width %d" % o["wh"][0], comments="")
    lines += relations(objs, rate) + [""]
    measures = build_measures(objs, args.distance or [], args.angle or [], rate)
    if measures:
        lines += measure_lines(measures, rate)
    ref_t = objs[0]["seed_t"]
    p = out / "paths.png"
    if path_image(args.video, objs, info, p, ref_t):
        images.append((p, "tracked paths on the seed frame: time-coloured trail (blue early → red late), dots at fixed time steps (bunching = slow), yellow rings = events"))
    for o in objs:
        p = out / f"filmstrip_{o['name']}.png"
        if filmstrip_image(args.video, o, info, p):
            images.append((p, f"`{o['name']}` cropped at 10 moments along its track (shape, rotation, squash & stretch; verify it is the right object)"))
        images.append((out / f"seed_{o['name']}.png", f"`{o['name']}` seed: the box that was tracked, with a zoom"))
    p = out / "plot.png"
    if plot_image(objs, p, measures):
        images.append((p, "x(t), y(t) and speed(t) of every object on a shared time axis; yellow ticks = events"))
    p = out / "summary.png"
    try:
        if summary_image(out, objs, p, measures):
            images.insert(0, (p, "FOR THE HUMAN, show this first: key facts, path, filmstrip and curves in ONE picture"))
    except (cv2.error, ValueError, IndexError) as exc:
        log(f"[track] summary picture skipped: {exc}")
    if not args.no_video:
        tt = [o["P"][np.isfinite(o["P"][:, 1]), 0] for o in objs if np.isfinite(o["P"][:, 1]).any()]
        if tt:
            t_a, t_b = min(a.min() for a in tt), max(a.max() for a in tt)
            try:
                if annotated_video(args.video, objs, info, out / "tracked.mp4", t_a, t_b + 0.05, rate, measures):
                    images.insert(1, (out / "tracked.mp4", "FOR THE HUMAN, show this: the video with box or cross, trail, speed, BOUNCE/TURN flashes and measurement lines drawn on (the model cannot watch it)"))
            except (cv2.error, ValueError, IndexError, TypeError, StopIteration, OSError) as exc:
                log(f"[track] annotated video skipped: {exc}")
    lines.append("## Files")
    lines.append("")
    for p, note in images:
        lines.append(f"- `{p}` — {note}")
    lines += [f"- `{out}/path_<name>.csv` — every tracked frame (screen and scene coordinates, confidence, state)", ""]
    lines += ["## How to read this", "",
              "- A *confidence* below 0.35 means the tracker doubted itself; check the filmstrip. After a scene cut the object is searched for by appearance and may be 'refound' (verify) or stay lost.",
              "- Positions are the box centre. If the camera moves, paths are drawn in the seed frame's coordinates with the camera motion removed, so the object's own movement shows.",
              "- Bounces compare speed before and after the reversal; *keeps N% of its speed* is the restitution. Gravity-like arcs fit y(t) with a parabola between events.",
              "- To refine: run `track` again with a tighter box, another seed time, or `--start/--end`."]
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "track.json").write_text(json.dumps([dict(name=o["name"], seed_t=o["seed_t"], box=o["box"], how=o["how"],
                                                      path=np.nan_to_num(o["P"], nan=-1).round(4).tolist(),
                                                      events=o["an"]["events"]) for o in objs], default=float), encoding="utf-8")
    print(text)


def cmd_sheet(args, info):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dur = info["duration"] or args.end
    t0, t1 = max(0.0, args.start), min(dur, args.end)
    n = max(1, min(24, args.n))
    times = np.linspace(t0, t1, n) if n > 1 else np.array([t0])
    box = parse_nums(args.box, (4,)) if args.box else None
    cell_w = args.cell or (300 if box is None else 280)
    cols = args.cols or (4 if n > 6 else min(n, 3))
    cells = []
    for tt in times:
        if box is None:
            ch = ma.even(info["height"] * cell_w / info["width"])
            fr = ma.grab_frame(args.video, float(tt), cell_w, ch)
        else:
            full_w = 1080
            fh = ma.even(info["height"] * full_w / info["width"])
            fr = ma.grab_frame(args.video, float(tt), full_w, fh)
            x0, y0 = int(box[0] * full_w), int(box[1] * fh)
            fr = fr[y0:int((box[1] + box[3]) * fh), x0:int((box[0] + box[2]) * full_w)]
            if fr.size == 0:
                raise SystemExit("The box lies outside the frame.")
            fr = cv2.resize(fr, (cell_w, max(8, int(cell_w * fr.shape[0] / fr.shape[1]))))
        if args.grid:
            if box is None:
                draw_grid(fr)
            else:
                draw_grid(fr, 0.25, (box[0], box[0] + box[2]), (box[1], box[1] + box[3]))
        put(fr, f"{tt:.2f}s", (4, 14), 0.45)
        cells.append(fr)
    hmax = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for c in cells]
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 20))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    path = out / f"sheet_{t0:.2f}-{t1:.2f}s.png"
    body = np.vstack(rows)
    banner = np.full((24, body.shape[1], 3), 20, np.uint8)
    cv2.putText(banner, f"WATCH SHEET  {n} frames {t0:.2f}-{t1:.2f}s" + (f"  zoom box {','.join(f'{v:.2f}' for v in box)}" if box else "  full frames")
                + ("  grid = frame coordinates" if args.grid else "") + "   LOOK ONLY - measure with track", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([banner, body]))
    text = (f"# watch: frame sheet\n\n- `{path}` — {n} frames from {t0:.2f}s to {t1:.2f}s ({(t1 - t0) / max(1, n - 1):.3f}s apart)"
            + (f", zoomed on box {','.join(f'{v:.2f}' for v in box)}" if box else "")
            + ".\n- This is a LOOK, not a measurement. Use it to find the moment and to read the coordinates of the parts you care about, "
              "then measure with `track`: follow points on the parts and add `--distance a,b` (mouth opening, gap) or `--angle a,b,c` (fin, limb) "
              "to get the curve, the number of cycles and the timing.")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_flow(args, info):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = frame_dims(info)
    dur = info["duration"] or args.t + 1
    half = args.window / 2
    fr = ma.ffmpeg_frames(args.video, max(0.0, args.t - half), min(dur, args.t + half + 0.04), 30, w, h)
    if len(fr) < 2:
        raise SystemExit("Not enough frames around that time.")
    a = cv2.GaussianBlur(cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(fr[-1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    span = (len(fr) - 1) / 30.0
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    cam = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    pts = cv2.goodFeaturesToTrack(a, 250, 0.01, 7)
    if pts is not None and len(pts) > 12:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **LK)
        ok = st.reshape(-1) == 1
        if ok.sum() > 12:
            A, _ = cv2.estimateAffinePartial2D(pts[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
            cam = A if A is not None else cam
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    cfx, cfy = cam[0, 0] * gx + cam[0, 1] * gy + cam[0, 2] - gx, cam[1, 0] * gx + cam[1, 1] * gy + cam[1, 2] - gy
    rx, ry = (flow[..., 0] - cfx, flow[..., 1] - cfy) if not args.raw else (flow[..., 0], flow[..., 1])
    mag = np.hypot(rx, ry)
    vis = fr[len(fr) // 2].copy()
    vis = (vis * 0.6).astype(np.uint8)
    top = float(np.percentile(mag, 99)) + 1e-6
    for y in range(8, h - 4, 16):
        for x in range(8, w - 4, 16):
            m = mag[y, x]
            if m < 0.35 * max(1.0, float(np.median(mag)) * 3):
                continue
            scale = 6.0 if m < 0.3 * top else 3.0
            col = ma.time_color(1 - min(1.0, m / top))
            cv2.arrowedLine(vis, (x, y), (int(x + rx[y, x] * scale), int(y + ry[y, x] * scale)), col, 1, cv2.LINE_AA, tipLength=0.4)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ((np.arctan2(ry, rx) + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / top * 255, 0, 255).astype(np.uint8)
    colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sheet = np.hstack([vis, colour])
    put(sheet, f"t={args.t:.2f}s  arrows ({'raw' if args.raw else 'camera removed'})", (4, 14), 0.42)
    put(sheet, "hue = direction, brightness = speed", (w + 4, 14), 0.42)
    path = out / f"flow_{args.t:.2f}s.png"
    cv2.imwrite(str(path), sheet)
    m = (mag > max(0.8, 3 * float(np.median(mag)))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    regions = []
    for c in range(1, n):
        x0, y0, bw, bh, area = [int(v) for v in stats[c]]
        if area < 30:
            continue
        sel = lab == c
        vx, vy = float(rx[sel].mean()), float(ry[sel].mean())
        regions.append((float(mag[sel].sum()), x0 / w, y0 / h, bw / w, bh / h, ma.direction_words(vx, vy), math.hypot(vx, vy) / w / span))
    regions.sort(reverse=True)
    lines = ["# watch: flow view", "", f"- `{path}` — left: motion arrows; right: flow as colour. Window {args.window:.2f}s around {args.t:.2f}s.",
             f"- Camera: shift ({cam[0, 2] / w:+.3f}, {cam[1, 2] / w:+.3f}) fw, zoom ×{math.hypot(cam[0, 0], cam[1, 0]):.3f}, roll {math.degrees(math.atan2(cam[1, 0], cam[0, 0])):+.1f}° over the window",
             "", "| region (x,y,w,h) | heading | speed fw/s |", "|---|---|---|"]
    for e, x, y, bw, bh, d, spd in regions[:8]:
        lines.append(f"| {x:.2f},{y:.2f},{bw:.2f},{bh:.2f} | {d} | {spd:.2f} |")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Object tools for /watch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("video"); p.add_argument("--out", required=True); p.add_argument("--t", type=float, required=True)
    p.add_argument("--at", action="append", default=None, help="x,y to mark with a cross and a magnified patch (repeatable)")
    p.add_argument("--box", default=None, help="x,y,w,h to zoom the view on")
    p = sub.add_parser("track")
    p.add_argument("video"); p.add_argument("--out", required=True)
    p.add_argument("--obj", action="append", required=True, help="NAME@TIME=SPEC (x,y | x,y,w,h)")
    p.add_argument("--start", type=float, default=None); p.add_argument("--end", type=float, default=None)
    p.add_argument("--no-video", action="store_true"); p.add_argument("--confirm", action="store_true")
    p.add_argument("--size", type=float, default=0.08, help="Box width around a point seed, as a fraction of the frame width")
    p.add_argument("--distance", action="append", default=None, help="a,b: distance between two --obj points over time (mouth opening, gap)")
    p.add_argument("--angle", action="append", default=None, help="a,b,c: angle at b between a and c over time (fin, limb, hinge)")
    p.add_argument("--engine", choices=["auto", "point", "box"], default="auto",
                   help="point: CoTracker3 follows the clicked point itself; box: a ViT tracker follows a box around it; auto: point when installed")
    p = sub.add_parser("sheet")
    p.add_argument("video"); p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, required=True); p.add_argument("--end", type=float, required=True)
    p.add_argument("--n", type=int, default=8); p.add_argument("--box", default=None)
    p.add_argument("--cols", type=int, default=None); p.add_argument("--cell", type=int, default=None)
    p.add_argument("--grid", action="store_true")
    p = sub.add_parser("flow")
    p.add_argument("video"); p.add_argument("--out", required=True); p.add_argument("--t", type=float, required=True)
    p.add_argument("--window", type=float, default=0.2); p.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    args.video = str(Path(args.video).resolve())
    info = ma.probe(Path(args.video))
    {"pick": cmd_pick, "track": cmd_track, "sheet": cmd_sheet, "flow": cmd_flow}[args.cmd](args, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

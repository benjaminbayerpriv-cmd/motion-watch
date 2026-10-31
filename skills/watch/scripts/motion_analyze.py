#!/usr/bin/env python3
"""Motion analysis engine for /watch --motion.

Runs inside the managed motion venv (numpy, opencv-python-headless, scipy); `motion.py`
launches it as a subprocess so the rest of the skill stays stdlib-only.

Instead of handing an agent dozens of stills, this measures the movement and renders a few
images that make it visible:

  * per-frame camera motion (pan / tilt / zoom / roll / shake) via RANSAC on tracked features
  * camera-compensated dense optical flow -> how much *things* move, where, and when
  * hard cuts, shot lengths, pacing
  * point tracks, clustered into movers, with speed profile and easing classification
  * audio onsets, tempo, loudness, and how well cuts / motion peaks line up with the beat
  * images: overview (thumbnails + curves), motion-history (colour = time), strobe, track paths

Everything is written to --out as report.md / report.json / *.png; the markdown is also printed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

ANALYSIS_W = 256     # grayscale width for feature tracking / camera model
FLOW_W = 160         # width for dense optical flow
DISP_W = 270         # width of frames used in rendered images
MAX_STEPS = 2400     # hard cap on analysed frames
BLOCKS = "▁▂▃▄▅▆▇█"
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


# ----------------------------------------------------------------------------- helpers

def fmt_t(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def smooth(x, n: int):
    x = np.asarray(x, dtype=np.float64)
    if n <= 1 or len(x) < 2:
        return x.copy()
    n = min(n, len(x))
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def sparkline(vals, lo=None, hi=None) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return ""
    lo = float(np.min(vals)) if lo is None else lo
    hi = float(np.max(vals)) if hi is None else hi
    if hi - lo < 1e-12:
        return BLOCKS[0] * len(vals)
    idx = np.clip(((vals - lo) / (hi - lo) * (len(BLOCKS) - 1)).round().astype(int), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[i] for i in idx)


def time_color(u: float):
    """Blue (early) -> cyan -> green -> yellow -> red (late), as a BGR tuple."""
    hue = int(round((1.0 - float(np.clip(u, 0, 1))) * 120))
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


_ASCII = {"\u00b0": " deg", "\u2192": "->", "\u2013": "-", "\u2014": "-", "\u00b2": "^2", "\u00b1": "+-", "\u00d7": "x",
          "\u2026": "...", "\u00b7": ".", "\u2022": "*", "\u2194": "<->"}


def ascii_safe(text: str) -> str:
    """OpenCV's Hershey font only draws ASCII; anything else corrupts the glyph advance."""
    for k, v in _ASCII.items():
        text = text.replace(k, v)
    return text.encode("ascii", "replace").decode()


def put_text(img, text, org, scale=0.45, color=(255, 255, 255), thick=1):
    text = ascii_safe(text)
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):   # outline: same stroke, same advance
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def where_words(x: float, y: float) -> str:
    h = "left" if x < 0.36 else "right" if x > 0.64 else "center"
    v = "top" if y < 0.36 else "bottom" if y > 0.64 else "middle"
    return f"{v}-{h}" if v != "middle" or h != "center" else "center"


def direction_words(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) < 1e-9:
        return "none"
    ang = math.degrees(math.atan2(-dy, dx)) % 360   # screen up = +90
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


# ----------------------------------------------------------------------------- pass 1

def probe(path: Path) -> dict:
    """Display-oriented size, fps and duration via ffprobe."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,r_frame_rate,duration:stream_tags=rotate:stream_side_data=rotation:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        st = j["streams"][0]
    except (ValueError, KeyError, IndexError):
        raise SystemExit(f"Cannot read video stream: {path}")

    def rate(v):
        try:
            a, b = str(v).split("/")
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return 0.0
    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate")) or 30.0
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 30.0
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float((st.get("tags") or {}).get("rotate", 0) or 0))
    if abs(rot) % 180 == 90:
        w, h = h, w
    dur = float(st.get("duration") or (j.get("format") or {}).get("duration") or 0)
    return dict(width=w, height=h, fps=fps, duration=dur)


def even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def ffmpeg_gray_stream(video, start, dur, rate, w, h):
    """Yield (w*h) grayscale uint8 frames sampled at `rate` fps."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-an", "-vf", f"fps={rate:.5f},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 4)
    n = w * h
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def pass1(video: Path, start: float, end: float | None, fps_cap: float, log):
    info = probe(video)
    fps = info["fps"]
    src_dur = info["duration"]
    end_t = min(end, src_dur) if end is not None and src_dur else (end if end is not None else src_dur)
    span = max(0.1, (end_t or 3600.0) - start)
    stride = max(1, int(round(fps / fps_cap)))
    if span * fps / stride > MAX_STEPS:
        stride = int(math.ceil(span * fps / MAX_STEPS))
    rate = fps / stride
    w0, h0 = info["width"], info["height"]
    width = ANALYSIS_W
    height = even(h0 * width / w0)
    fw_h = even(h0 * FLOW_W / w0)
    gx, gy = np.meshgrid(np.arange(FLOW_W, dtype=np.float32), np.arange(fw_h, dtype=np.float32))

    S = {k: [] for k in ("t", "fidx", "cut", "shot", "mag", "res", "loc", "area", "cx", "cy", "vx", "vy",
                         "tx", "ty", "zoom", "rot", "inl", "bright", "diff", "M")}
    tracks: dict[int, dict] = {}
    blobs_all: list = []
    next_id = 0
    pts = np.zeros((0, 1, 2), np.float32)
    ids: list[int] = []
    prev_gray = prev_flow = prev_thumb = prev_hist = None
    M = np.eye(3)
    shot = 0
    last_cut_step = -99
    last_detect = 0
    i = 0

    def detect(gray, existing, step):
        mask = np.full(gray.shape, 255, np.uint8)
        for p in existing.reshape(-1, 2):
            cv2.circle(mask, (int(p[0]), int(p[1])), 8, 0, -1)
        want = 260 - len(existing)
        if want < 20:
            return np.zeros((0, 1, 2), np.float32)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=want, qualityLevel=0.01, minDistance=7,
                                      blockSize=7, mask=mask)
        return np.zeros((0, 1, 2), np.float32) if new is None else new.astype(np.float32)

    def register(new, step):
        nonlocal next_id, pts, ids
        if len(new) == 0:
            return
        inv = np.linalg.inv(M)
        for p in new.reshape(-1, 2):
            s = inv @ np.array([p[0], p[1], 1.0])
            tracks[next_id] = {"shot": shot, "f": [step], "p": [(s[0], s[1])]}
            ids.append(next_id)
            next_id += 1
        pts = np.concatenate([pts, new], axis=0) if len(pts) else new

    for raw in ffmpeg_gray_stream(video, start, span if end is not None else 0, rate, width, height):
        fidx = int(round((start + i / rate) * fps))
        gray = cv2.GaussianBlur(raw, (3, 3), 0)
        fgray = cv2.resize(gray, (FLOW_W, gx.shape[0]), interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(gray, (64, max(8, int(64 * height / width))), interpolation=cv2.INTER_AREA).astype(np.float32)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        bright = float(gray.mean()) / 255.0

        blobs = []
        row = dict(cut=False, mag=0.0, res=0.0, loc=0.0, area=0.0, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
                   tx=0.0, ty=0.0, zoom=0.0, rot=0.0, inl=1.0, diff=0.0)
        if prev_gray is not None:
            d = float(np.abs(thumb - prev_thumb).mean())
            hc = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
            row["diff"] = d / 255.0
            is_cut = (d > 30 or (hc < 0.5 and d > 18)) and (i - last_cut_step) >= 3
            if is_cut:
                shot += 1
                last_cut_step = i
                row["cut"] = True
                M = np.eye(3)
                pts = np.zeros((0, 1, 2), np.float32)
                ids = []
                register(detect(gray, pts, i), i)
                last_detect = i
            else:
                A = None
                if len(pts) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
                    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK)
                    fb = np.linalg.norm((p0b - pts).reshape(-1, 2), axis=1)
                    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
                    inside = ((p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < width) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < height))
                    good &= inside
                    keep_ids = [ids[j] for j in range(len(ids)) if good[j]]
                    a, b = pts[good], p1[good]
                    if len(a) >= 8:
                        A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
                        row["inl"] = float(inl.mean()) if inl is not None else 0.0
                    pts, ids = p1[good].copy(), keep_ids
                if A is None:
                    A = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
                    if len(pts) >= 8:
                        row["inl"] = 0.0
                A3 = np.vstack([A, [0, 0, 1]])
                M = A3 @ M
                inv = np.linalg.inv(M)
                for p, tid in zip(pts.reshape(-1, 2), ids):
                    s = inv @ np.array([p[0], p[1], 1.0])
                    tr = tracks[tid]
                    tr["f"].append(i)
                    tr["p"].append((s[0], s[1]))
                scale = math.sqrt(max(1e-9, A[0, 0] ** 2 + A[1, 0] ** 2))
                c = np.array([width / 2, height / 2, 1.0])
                shift = A3 @ c - c
                row["tx"] = float(shift[0]) / width
                row["ty"] = float(shift[1]) / width
                row["zoom"] = math.log(scale)
                row["rot"] = math.atan2(A[1, 0], A[0, 0])
                # dense flow, compared with what the camera model alone would predict
                flow = cv2.calcOpticalFlowFarneback(prev_flow, fgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                kf = FLOW_W / width
                cx_f = gx - (A[0, 0] * gx + A[0, 1] * gy + A[0, 2] * kf)
                cy_f = gy - (A[1, 0] * gx + A[1, 1] * gy + A[1, 2] * kf)
                rx, ry = flow[..., 0] + cx_f, flow[..., 1] + cy_f
                tot = np.hypot(flow[..., 0], flow[..., 1])
                res = np.hypot(rx, ry)
                row["mag"] = float(tot.mean()) / FLOW_W
                row["res"] = float(res.mean()) / FLOW_W
                m = res > 0.5
                row["area"] = float(m.mean())
                kk = max(4, int(res.size * 0.015))
                row["loc"] = float(np.partition(res.ravel(), -kk)[-kk:].mean()) / FLOW_W
                blobs = find_blobs(res)
                if m.sum() > 12:
                    wgt = (res * m)
                    wsum = float(wgt.sum())
                    row["cx"] = float((wgt * gx).sum() / wsum) / FLOW_W
                    row["cy"] = float((wgt * gy).sum() / wsum) / gx.shape[0]
                    row["vx"] = float((rx * m).sum() / m.sum()) / FLOW_W
                    row["vy"] = float((ry * m).sum() / m.sum()) / FLOW_W
                if len(pts) < 60 or i - last_detect >= 20:
                    register(detect(gray, pts, i), i)
                    last_detect = i
        else:
            register(detect(gray, pts, i), i)
            last_detect = i

        blobs_all.append(blobs)
        S["t"].append(start + i / rate)
        S["fidx"].append(fidx)
        S["shot"].append(shot)
        S["bright"].append(bright)
        S["M"].append(M.copy())
        for key, val in row.items():
            S[key].append(val)
        prev_gray, prev_flow, prev_thumb, prev_hist = gray, fgray, thumb, hist
        i += 1
        if i % 300 == 0:
            log(f"[motion] analysed {i} frames ({fmt_t(start + i / rate)})")
    if i < 3:
        raise SystemExit("Video has too few readable frames for motion analysis.")
    out = {k: np.array(v) for k, v in S.items() if k != "M"}
    out["M"] = np.array(S["M"])
    out["blobs"] = blobs_all
    out["cut"] = out["cut"].astype(bool)
    return out, tracks, dict(fps=fps, dt=1.0 / rate, rate=rate, stride=stride, steps=i, width=w0, height=h0,
                             aw=width, ah=height, start=start)


# ----------------------------------------------------------------------------- derived data

def find_blobs(res, top=3):
    """Connected regions of camera-compensated motion: [(x, y, energy, area)] in flow-pixel units."""
    thr = max(0.6, 4.0 * float(np.median(res)))
    m = (res > thr).astype(np.uint8)
    if m.sum() < 8:
        return []
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    energy = np.bincount(lab.ravel(), weights=res.ravel(), minlength=n)
    out = []
    for c in range(1, n):
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area < 10 or area > 0.45 * m.size:
            continue
        out.append((float(cent[c, 0]), float(cent[c, 1]), float(energy[c]), area))
    out.sort(key=lambda b: -b[2])
    return out[:top]


def link_blobs(S, meta):
    """Follow moving regions through each shot -> paths in the shot's stabilised coordinates."""
    W = meta["aw"]
    k = W / FLOW_W
    max_jump, max_gap = 0.20 * W, 3
    n = len(S["t"])
    active, done = [], []
    shot_now = -1
    for i in range(n):
        if S["shot"][i] != shot_now:
            done += active
            active, shot_now = [], S["shot"][i]
        inv = np.linalg.inv(S["M"][i])
        cands = []
        for (x, y, e, a) in S["blobs"][i]:
            q = inv @ np.array([x * k, y * k, 1.0])
            cands.append((q[:2], e, a))
        used = set()
        for tr in sorted(active, key=lambda t: -t["energy"]):
            pred = tr["p"][-1] + tr["v"] * (i - tr["f"][-1])
            best, bd = None, max_jump
            for j, (q, e, a) in enumerate(cands):
                if j in used:
                    continue
                d = float(np.hypot(*(q - pred)))
                if d < bd:
                    best, bd = j, d
            if best is not None:
                used.add(best)
                q, e, a = cands[best]
                gap = i - tr["f"][-1]
                tr["v"] = 0.6 * tr["v"] + 0.4 * (q - tr["p"][-1]) / gap
                tr["f"].append(i)
                tr["p"].append(q)
                tr["energy"] = 0.8 * tr["energy"] + 0.2 * e
        for j, (q, e, a) in enumerate(cands):
            if j not in used:
                active.append(dict(f=[i], p=[q], v=np.zeros(2), energy=e, shot=S["shot"][i]))
        still = []
        for tr in active:
            (still if i - tr["f"][-1] <= max_gap else done).append(tr)
        active = still
    done += active
    out = []
    for tr in done:
        f, p = np.array(tr["f"]), np.array(tr["p"])
        if len(f) < 8:
            continue
        full = np.arange(f[0], f[-1] + 1)
        pos = np.c_[np.interp(full, f, p[:, 0]), np.interp(full, f, p[:, 1])]
        pos = np.c_[smooth(pos[:, 0], 3), smooth(pos[:, 1], 3)]
        length = float(np.hypot(*np.diff(pos, axis=0).T).sum())
        net = float(np.hypot(*(pos[-1] - pos[0])))
        if net < 0.05 * W or net / max(length, 1e-9) < 0.25:
            continue
        out.append(dict(shot=tr["shot"], frames=full, pos=pos, members=1, kind="blob"))
    return out


def finish_movers(raw, S, meta):
    dt, W = meta["dt"], meta["aw"]
    result = []
    for c in raw:
        pos = c["pos"]
        step = np.hypot(*np.diff(pos, axis=0).T)
        sp = smooth(step / dt / W, 3)
        j = int(sp.argmax())
        net = pos[-1] - pos[0]
        f = c["frames"]
        scr = S["M"][f[len(f) // 2]] @ np.array([pos[len(f) // 2, 0], pos[len(f) // 2, 1], 1.0])
        result.append(dict(shot=c["shot"], f0=int(f[0]), f1=int(f[-1]), t0=float(S["t"][f[0]]), t1=float(S["t"][f[-1]]),
                           seed_t=float(S["t"][f[len(f) // 2]]), seed_xy=(float(scr[0] / W), float(scr[1] / meta["ah"])),
                           pos=pos, frames=f, length_w=float(step.sum()) / W,
                           net_dir=direction_words(net[0], net[1]),
                           start_xy=(float(pos[0, 0] / W), float(pos[0, 1] / meta["ah"])),
                           end_xy=(float(pos[-1, 0] / W), float(pos[-1, 1] / meta["ah"])),
                           peak_speed=float(sp.max()), peak_t=float(S["t"][f[min(j + 1, len(f) - 1)]]),
                           speed=sp, easing=classify_easing(pos, dt), members=c.get("members", 1), kind=c.get("kind", "points")))
    # drop duplicates (the same object found by both trackers) and negligible wiggles
    result = [r for r in result if r["length_w"] >= 0.12 and r["peak_speed"] >= 0.08]
    result.sort(key=lambda r: -r["length_w"])
    keep = []
    for r in result:
        dup = False
        for q in keep:
            if q["shot"] != r["shot"]:
                continue
            common = np.intersect1d(q["frames"], r["frames"])
            if len(common) < 4:
                continue
            a = q["pos"][np.searchsorted(q["frames"], common)]
            b = r["pos"][np.searchsorted(r["frames"], common)]
            if np.median(np.hypot(*(a - b).T)) < 0.12 * W:
                dup = True
                break
        if not dup:
            keep.append(r)
    for n, r in enumerate(keep[:6]):
        r["tag"] = chr(ord("A") + n)
    return keep


def build_shots(S, dt):
    shot = S["shot"]
    shots = []
    start = 0
    for j in range(1, len(shot) + 1):
        if j == len(shot) or shot[j] != shot[start]:
            shots.append((start, j - 1))
            start = j
    return shots


def camera_label(S, i0, i1, dt):
    if i1 - i0 < 3:
        return "too short", {}
    sl = slice(i0 + 1, i1 + 1)
    tx, ty, zm, rot = S["tx"][sl], S["ty"][sl], S["zoom"][sl], S["rot"][sl]
    net_x, net_y = float(tx.sum()), float(ty.sum())          # content shift, fraction of width
    net_zoom = float(zm.sum())                               # log scale
    net_rot = math.degrees(float(rot.sum()))
    hp_x = tx - smooth(tx, 7)
    hp_y = ty - smooth(ty, 7)
    jitter = float(np.sqrt(np.mean(hp_x ** 2 + hp_y ** 2)))
    parts = []
    if abs(net_x) > 0.06:
        parts.append(f"pan {'left' if net_x > 0 else 'right'}")      # content moves right => camera pans left
    if abs(net_y) > 0.06:
        parts.append(f"tilt {'up' if net_y > 0 else 'down'}")
    if abs(net_zoom) > 0.06:
        parts.append("zoom/push in" if net_zoom > 0 else "zoom/pull out")
    if abs(net_rot) > 8:
        parts.append(f"roll {net_rot:+.0f}°")
    if jitter > 0.012:
        parts.append("strong shake")
    elif jitter > 0.004:
        parts.append("handheld")
    label = ", ".join(parts) if parts else "static"
    return label, dict(net_x=net_x, net_y=net_y, net_zoom_pct=(math.exp(net_zoom) - 1) * 100,
                       net_roll_deg=net_rot, jitter=jitter)


EASINGS = {
    "linear": lambda u: u,
    "ease-in (quad)": lambda u: u ** 2,
    "ease-in (cubic)": lambda u: u ** 3,
    "ease-out (quad)": lambda u: 1 - (1 - u) ** 2,
    "ease-out (cubic)": lambda u: 1 - (1 - u) ** 3,
    "ease-in-out (smoothstep)": lambda u: 3 * u ** 2 - 2 * u ** 3,
    "ease-in-out (cubic)": lambda u: np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2),
}


def classify_easing(pos, dt):
    """pos: (n,2) positions of one mover. Fits its main move to standard easing curves.

    Start and duration are free parameters, so a slow tail that fell below the motion threshold
    (typical for ease-out) does not distort the classification.
    """
    n = len(pos)
    if n < 6:
        return dict(label="too short")
    vel = np.diff(pos, axis=0) / dt
    speed = smooth(np.hypot(vel[:, 0], vel[:, 1]), 3)
    vmax = float(speed.max())
    if vmax < 1e-9:
        return dict(label="still")
    thr = 0.05 * vmax
    best, cur = (0, 0), None
    for j, v in enumerate(speed):
        if v > thr:
            cur = (cur[0], j) if cur else (j, j)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    a, b = best
    if b - a < 4:
        return dict(label="too short")
    a, b = max(0, a - 2), min(n - 2, b + 2)
    seg = pos[a:b + 2]
    net = seg[-1] - seg[0]
    net_len = float(np.hypot(*net))
    path_len = float(np.hypot(*np.diff(seg, axis=0).T).sum())
    if net_len < 1e-9 or net_len / max(path_len, 1e-9) < 0.6:
        return dict(label="curved / oscillating path", straightness=net_len / max(path_len, 1e-9))
    d = net / net_len
    s = ((seg - seg[0]) @ d) / net_len
    tau = np.arange(len(s)) * dt
    T = float(tau[-1])
    peak = float(s.max())
    fits = {}
    for name, fn in EASINGS.items():
        best_err, best_par = 9.0, None
        for t0 in np.linspace(-0.25 * T, 0.35 * T, 13):
            for D in np.linspace(0.35 * T, 1.7 * T, 28):
                m = np.clip(fn(np.clip((tau - t0) / D, 0, 1)), 0, 1)
                amp = float(np.clip((m @ s) / max(m @ m, 1e-9), 0.85, 1.15))
                err = float(np.sqrt(np.mean((s - amp * m) ** 2)))
                if err < best_err:
                    best_err, best_par = err, (t0, D)
        fits[name] = (best_err, best_par)
    name = min(fits, key=lambda k: fits[k][0])
    err, (t0, D) = fits[name]
    label = name
    near = [k for k in fits if fits[k][0] <= err + 0.004]
    if len(near) > 1 and len({k.split(" (")[0] for k in near}) == 1:   # same family, different strength
        label = near[0].split(" (")[0] + " (" + "/".join(k.split(" (")[1].rstrip(")") for k in near) + ", strength not distinguishable)"
    if peak > 1.06:
        label = f"overshoot / spring (+{(peak - 1) * 100:.0f}% past target) after {label}"
    return dict(label=label, rmse=err, move_start=a, move_end=b + 1, duration=float(min(D, T * 1.2)),
                straightness=net_len / path_len, scores={k: round(v[0], 3) for k, v in fits.items()})


def cluster_movers(tracks, S, meta):
    """Group feature tracks that move relative to the scene into per-object paths."""
    dt = meta["dt"]
    W = meta["aw"]
    movers_by_shot: dict[int, list] = {}
    for tid, tr in tracks.items():
        if len(tr["f"]) < 6:
            continue
        p = np.array(tr["p"])
        net = np.hypot(*(p[-1] - p[0]))
        step = np.hypot(*np.diff(p, axis=0).T)
        if net < 0.035 * W or step.mean() > 0.12 * W:
            continue
        # a real mover travels coherently: path length close to net displacement
        if net / max(step.sum(), 1e-9) < 0.35:
            continue
        movers_by_shot.setdefault(tr["shot"], []).append((tid, np.array(tr["f"]), p, float(step.sum())))
    clusters = []
    for shot, items in movers_by_shot.items():
        items.sort(key=lambda it: -it[3])
        local = []
        for tid, f, p, ln in items:
            placed = False
            for c in local:
                common = np.intersect1d(f, c["frames"])
                if len(common) < 3:
                    continue
                pa = p[np.searchsorted(f, common)]
                pb = c["pos"][np.searchsorted(c["frames"], common)]
                if np.median(np.hypot(*(pa - pb).T)) < 0.10 * W:
                    c["members"].append((f, p))
                    placed = True
                    break
            if not placed:
                local.append(dict(shot=shot, frames=f, pos=p, members=[(f, p)]))
        for c in local:
            if len(c["members"]) < 2 and len(c["frames"]) < 10:
                continue
            f0 = min(m[0][0] for m in c["members"])
            f1 = max(m[0][-1] for m in c["members"])
            n = f1 - f0 + 1
            vel = np.full((n, 2), np.nan)
            for j in range(n - 1):
                vs = []
                for f, p in c["members"]:
                    q = np.searchsorted(f, f0 + j)
                    if q + 1 < len(f) and f[q] == f0 + j and f[q + 1] == f0 + j + 1:
                        vs.append(p[q + 1] - p[q])
                if vs:
                    vel[j] = np.median(np.array(vs), axis=0)
            valid = ~np.isnan(vel[:, 0])
            if valid.sum() < 6:
                continue
            vel[~valid] = 0.0
            first_pos = np.median([p[0] for f, p in c["members"] if f[0] == f0] or [c["members"][0][1][0]], axis=0)
            path = np.vstack([first_pos, first_pos + np.cumsum(vel[:-1], axis=0)])
            c["frames"], c["pos"] = np.arange(f0, f1 + 1), path
            clusters.append(c)
    for c in clusters:
        c["members"] = len(c["members"])
        c["kind"] = "points"
    return clusters


# ----------------------------------------------------------------------------- audio

def analyse_audio(video: Path, start: float, end: float | None, cut_times, peak_times, duration, log):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(video), "-vn", "-ac", "1", "-ar", "22050", "-f", "f32le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    y = np.frombuffer(raw, dtype=np.float32)
    sr = 22050
    if len(y) < sr // 2 or float(np.abs(y).max()) < 1e-4:
        return None
    n_fft, hop = 1024, 512
    nfr = 1 + (len(y) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(y, (nfr, n_fft), (y.strides[0] * hop, y.strides[0]))
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    logspec = np.log1p(10 * spec)
    flux = np.maximum(0, np.diff(logspec, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    flux = flux / (np.median(flux) + 1e-9)
    env_t = start + (np.arange(nfr) * hop + n_fft / 2) / sr
    senv = smooth(flux, 3)
    hi = float(np.median(senv) + 1.2 * senv.std())
    pk, _ = signal.find_peaks(senv, height=hi, distance=max(1, int(0.09 * sr / hop)))
    onsets = env_t[pk]
    # loudness
    win = sr // 4
    nw = len(y) // win
    rms = np.sqrt((y[:nw * win].reshape(nw, win) ** 2).mean(axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)
    rms_t = start + (np.arange(nw) + 0.5) * win / sr
    # tempo
    tempo = None
    e = senv - senv.mean()
    if len(e) > 200:
        ac = np.correlate(e, e, mode="full")[len(e) - 1:]
        lo, hi_l = int(0.30 * sr / hop), int(1.0 * sr / hop)
        if hi_l < len(ac) - 1:
            seg = ac[lo:hi_l]
            j = int(seg.argmax()) + lo
            tempo = dict(bpm=60.0 * sr / hop / j, confidence=float(ac[j] / (ac[0] + 1e-9)))
    out = dict(onsets=onsets, env_t=env_t, env=senv, rms_t=rms_t, rms_db=rms_db, tempo=tempo,
               onset_rate=len(onsets) / max(duration, 1e-6))
    out["cut_sync"] = sync_score(cut_times, onsets, duration, start)
    out["peak_sync"] = sync_score(peak_times, onsets, duration, start)
    return out


def sync_score(events, onsets, duration, t0, tol=0.08):
    events = np.asarray(events, dtype=np.float64)
    if len(events) < 3 or len(onsets) < 4:
        return None
    def hit(ev):
        d = np.abs(ev[:, None] - onsets[None, :])
        return float((d.min(axis=1) <= tol).mean())
    obs = hit(events)
    rng = np.random.default_rng(1)
    null = []
    for _ in range(300):
        sh = rng.uniform(0, duration)
        null.append(hit(t0 + (events - t0 + sh) % duration))
    chance = float(np.mean(null))
    nearest = onsets[np.abs(events[:, None] - onsets[None, :]).argmin(axis=1)]
    return dict(hit_rate=obs, chance=chance, ratio=obs / chance if chance > 0 else None,
                median_offset_ms=float(np.median(nearest - events) * 1000), n=len(events))


# ----------------------------------------------------------------------------- pass 2: images

def disp_size(meta):
    return DISP_W, even(meta["height"] * DISP_W / meta["width"])


def ffmpeg_frames(video, t0, t1, rate, dw, dh):
    """BGR frames sampled at `rate` fps from [t0, t1), scaled to dw x dh. Returns (n, dh, dw, 3)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-i", str(video), "-t", f"{max(0.05, t1 - t0):.3f}",
           "-an", "-vf", f"fps={rate:.5f},scale={dw}:{dh}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    n = len(raw) // (dw * dh * 3)
    return np.frombuffer(raw[:n * dw * dh * 3], np.uint8).reshape(n, dh, dw, 3)


def grab_frame(video, t, dw, dh):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1", "-an",
           "-vf", f"scale={dw}:{dh}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    if len(raw) < dw * dh * 3:
        return np.zeros((dh, dw, 3), np.uint8)
    return np.frombuffer(raw[:dw * dh * 3], np.uint8).reshape(dh, dw, 3).copy()


def stabilise(frame, M_rel, k):
    """Warp a display frame into the coordinates of the window's first frame."""
    Sk = np.diag([k, k, 1.0])
    Mi = Sk @ np.linalg.inv(M_rel) @ np.linalg.inv(Sk)
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, Mi[:2], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stable_window(S, a, b, meta, max_drift=0.12):
    """Longest window of shot [a, b] whose camera drift stays small, preferring lively ones.

    Drift = how far (fraction of frame width) the image corners wander from where they were at the
    window start. Panning shots would smear a whole-shot overlay, so they get their best short window.
    """
    W, H, dt = meta["aw"], meta["ah"], meta["dt"]
    corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], float).T
    Ms, energy = S["M"], S["loc"]
    csum = np.concatenate([[0], np.cumsum(energy)])

    def drift(j0, j1):
        rel = Ms[j0:j1 + 1] @ np.linalg.inv(Ms[j0])
        disp = np.linalg.norm((rel @ corners)[:, :2, :] - corners[None, :2, :], axis=1)
        return float(disp.max()) / W

    n = b - a + 1
    lengths = sorted({n, *[min(n, max(6, int(sec / dt))) for sec in (2.5, 1.5, 0.9)]}, reverse=True)
    best_unstable = None
    for L in lengths:
        starts = list(range(a, b - L + 2, max(1, L // 8)))
        starts.sort(key=lambda j: -(csum[j + L] - csum[j]))
        for j in starts[:14]:
            d = drift(j, j + L - 1)
            if best_unstable is None or d < best_unstable[2]:
                best_unstable = (j, j + L - 1, d, L == n)
            if d <= max_drift:
                return dict(j0=j, j1=j + L - 1, drift=d, full=(L == n), stable=True)
    j0, j1, d, full = best_unstable
    return dict(j0=j0, j1=j1, drift=d, full=full, stable=False)


def window_frames(video, S, win, meta, limit=140):
    dw, dh = disp_size(meta)
    t0, t1 = float(S["t"][win["j0"]]), float(S["t"][win["j1"]] + meta["dt"])
    fr = ffmpeg_frames(video, t0, t1, meta["rate"], dw, dh)
    steps = list(range(win["j0"], win["j0"] + len(fr)))
    sel = list(range(len(fr)))
    if len(sel) > limit:
        sel = [int(x) for x in np.linspace(0, len(fr) - 1, limit)]
    k = dw / meta["aw"]
    M0inv = np.linalg.inv(S["M"][win["j0"]])
    return [(steps[q], stabilise(fr[q], S["M"][steps[q]] @ M0inv, k)) for q in sel if steps[q] <= win["j1"]]


def render_history(sf, S, path, label):
    if len(sf) < 4:
        return False
    imgs = np.stack([f for _, f in sf])
    bg = np.median(imgs, axis=0).astype(np.uint8)
    canvas = (cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.55).astype(np.uint8)
    n = len(sf)
    prev = cv2.GaussianBlur(cv2.cvtColor(sf[0][1], cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    diffs = []
    for _, f in sf[1:]:
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
        diffs.append(np.abs(g - prev))
        prev = g
    scale = max(10.0, float(np.percentile(np.concatenate([d.ravel() for d in diffs]), 99)) * 0.25)
    for j, d in enumerate(diffs):
        m = (d > scale).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        col = np.array(time_color(j / max(1, n - 2)), np.float32)
        a = (m[..., None] * 0.8).astype(np.float32)
        canvas = (canvas * (1 - a) + col * a).astype(np.uint8)
    return finish_with_legend(canvas, path, f"{label} colour=time",
                              S["t"][sf[0][0]], S["t"][sf[-1][0]])


def finish_with_legend(canvas, path, title, t0, t1):
    h, w = canvas.shape[:2]
    bar = np.zeros((26, w, 3), np.uint8)
    for x in range(w):
        bar[:, x] = time_color(x / max(1, w - 1))
    out = np.vstack([canvas, bar])
    out[:22] = (out[:22] * 0.35).astype(np.uint8)
    cv2.putText(out, ascii_safe(title), (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    put_text(out, fmt_t(t0), (4, h + 18), 0.4, (0, 0, 0))
    txt = fmt_t(t1)
    put_text(out, txt, (w - 52, h + 18), 0.4, (0, 0, 0))
    return cv2.imwrite(str(path), out)


def render_strobe(sf, S, path, label, k_frames=7):
    if len(sf) < 6:
        return False
    imgs = np.stack([f for _, f in sf])
    bg = np.median(imgs, axis=0).astype(np.uint8)
    energy = np.array([S["loc"][s] for s, _ in sf])
    live = np.where(energy > 0.3 * energy.max())[0] if energy.max() > 0 else np.arange(len(sf))
    if len(live) < k_frames:
        live = np.arange(len(sf))
    pick = [live[int(x)] for x in np.linspace(0, len(live) - 1, k_frames)]
    canvas = bg.copy()
    bgg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
    for rank, j in enumerate(pick):
        f = sf[j][1]
        d = np.abs(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) - bgg)
        m = (d > 26).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        if m.sum() < 30:
            continue
        alpha = 0.55 + 0.45 * rank / max(1, len(pick) - 1)
        a = (m[..., None] * alpha).astype(np.float32)
        canvas = (canvas * (1 - a) + f * a).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = time_color(rank / max(1, len(pick) - 1))
        cv2.drawContours(canvas, cnts, -1, col, 1)
        ys, xs = np.nonzero(m)
        put_text(canvas, f"{S['t'][sf[j][0]]:.1f}s", (int(np.clip(xs.mean() - 12, 0, DISP_W - 40)), int(np.clip(ys.min() + 10, 10, canvas.shape[0] - 4))), 0.36, col)
    return finish_with_legend(canvas, path, f"{label} strobe", S["t"][sf[pick[0]][0]], S["t"][sf[pick[-1]][0]])


def render_tracks(sf, S, win, movers, meta, path, label):
    if len(sf) < 4:
        return False
    canvas = (sf[0][1] * 0.6).astype(np.uint8)
    dw, dh = disp_size(meta)
    k = dw / meta["aw"]
    M0 = S["M"][win["j0"]]
    every = max(1, int(round(0.1 / meta["dt"])))         # a dot every 0.1 s: dense dots = slow, sparse = fast
    drawn = 0
    t_lo, t_hi = S["t"][win["j1"]], S["t"][win["j0"]]
    for mv in movers:
        if "tag" not in mv:
            continue
        sel = np.where((mv["frames"] >= win["j0"]) & (mv["frames"] <= win["j1"]))[0]
        if len(sel) < 4:
            continue
        hom = np.c_[mv["pos"][sel], np.ones(len(sel))]
        pos = (hom @ M0.T)[:, :2] * k
        for j in range(len(pos) - 1):
            cv2.line(canvas, tuple(int(v) for v in pos[j]), tuple(int(v) for v in pos[j + 1]),
                     time_color(j / max(1, len(pos) - 2)), 2, cv2.LINE_AA)
        for j in range(0, len(pos), every):
            cv2.circle(canvas, tuple(int(v) for v in pos[j]), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(int(v) for v in pos[-3]), tuple(int(v) for v in pos[-1]), (255, 255, 255), 2,
                        cv2.LINE_AA, tipLength=0.9)
        put_text(canvas, mv["tag"], (int(pos[0, 0]) + 5, int(pos[0, 1]) - 5), 0.5, (0, 255, 255), 2)
        t_lo, t_hi = min(t_lo, S["t"][mv["frames"][sel[0]]]), max(t_hi, S["t"][mv["frames"][sel[-1]]])
        drawn += 1
    if not drawn:                                         # nothing trackable: show where the motion energy travels
        steps = np.arange(win["j0"], win["j1"] + 1)
        live = steps[S["loc"][steps] > 0.25 * max(S["loc"][steps].max(), 1e-9)]
        if len(live) < 4:
            return False
        pos = np.c_[S["cx"][live] * dw, S["cy"][live] * dh]
        for j in range(len(pos) - 1):
            cv2.line(canvas, tuple(int(v) for v in pos[j]), tuple(int(v) for v in pos[j + 1]),
                     time_color(j / max(1, len(pos) - 2)), 2, cv2.LINE_AA)
        put_text(canvas, "energy centre", (6, 34), 0.4, (0, 255, 255))
        t_lo, t_hi = S["t"][live[0]], S["t"][live[-1]]
    return finish_with_legend(canvas, path, f"{label} paths, dot=0.1s", t_lo, t_hi)


def render_overview(video, S, shots, peaks, audio, meta, path):
    W = 1200
    L, R = 58, 14
    n_th = min(10, len(shots))
    pick = [shots[int(x)] for x in np.linspace(0, len(shots) - 1, n_th)] if len(shots) > n_th else shots
    mids = [(a + b) // 2 for a, b in pick]
    tw = (W - L - R) // max(1, n_th)
    thumbs = []
    thw = tw - 4
    thh = min(300, even(meta["height"] * thw / meta["width"]))
    for (a, b), m in zip(pick, mids):
        fr = grab_frame(video, float(S["t"][m]), thw, thh)
        put_text(fr, f"{S['t'][a]:.1f}-{S['t'][b]:.1f}s", (3, 13), 0.36)
        thumbs.append(fr)
    th_h = max(t.shape[0] for t in thumbs)
    strip = np.zeros((th_h + 8, W, 3), np.uint8)
    for j, t in enumerate(thumbs):
        strip[4:4 + t.shape[0], L + j * tw + 2:L + j * tw + 2 + t.shape[1]] = t
    put_text(strip, "shots", (4, 20), 0.4)

    t = S["t"]
    t0, t1 = float(t[0]), float(t[-1])
    span = max(t1 - t0, 1e-6)
    X = lambda v: int(L + (v - t0) / span * (W - L - R))
    panels = []

    def panel(title, series, colors, ymax=None, ticks=None, h=112):
        img = np.full((h, W, 3), 22, np.uint8)
        cv2.rectangle(img, (L, 4), (W - R, h - 6), (60, 60, 60), 1)
        put_text(img, title, (L + 4, 16), 0.38)
        top = ymax if ymax else max((float(np.max(s[1])) for s in series if len(s[1])), default=1.0) or 1.0
        for xs, ys, col in zip([s[0] for s in series], [s[1] for s in series], colors):
            pts = np.array([[X(a), int(h - 8 - np.clip(b / top, 0, 1) * (h - 28))] for a, b in zip(xs, ys)], np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        for cx in S["t"][S["cut"]]:
            cv2.line(img, (X(cx), 5), (X(cx), h - 7), (60, 60, 220), 1)
        if ticks is not None:
            for tk in ticks:
                cv2.line(img, (X(tk), h - 14), (X(tk), h - 7), (0, 220, 255), 1)
        return img

    dt = meta["dt"]
    mot = smooth(S["mag"], 3) / dt
    p_t = [S["t"][p] for p in peaks]
    pm = panel("total motion (frame widths / s)  red lines = cuts, yellow ticks = audio onsets", [(t, mot, (80, 220, 80))],
               [(80, 220, 80)], ticks=audio["onsets"] if audio else None)
    for p in peaks:
        cv2.circle(pm, (X(S["t"][p]), int(pm.shape[0] - 8 - np.clip(mot[p] / (mot.max() or 1), 0, 1) * (pm.shape[0] - 28))), 4, (255, 255, 255), 1)
    panels.append(pm)
    obj = smooth(S["loc"], 3) / dt
    obj_mean = smooth(S["res"], 3) / dt
    panels.append(panel("object motion, camera removed: strongest 1.5% of pixels (orange) / whole-frame mean (dim)",
                        [(t, obj_mean * 4, (110, 80, 40)), (t, obj, (255, 160, 60))], [(110, 80, 40), (255, 160, 60)]))
    cam_t = smooth(np.hypot(S["tx"], S["ty"]), 3) / dt
    cam_z = smooth(np.abs(S["zoom"]), 3) / dt
    panels.append(panel("camera: translation (cyan) / zoom (magenta)", [(t, cam_t, (255, 220, 60)), (t, cam_z * 3, (220, 60, 220))],
                        [(255, 220, 60), (220, 60, 220)]))
    if audio:
        db = np.clip((audio["rms_db"] + 60) / 60, 0, 1)
        env = np.clip(audio["env"] / (np.percentile(audio["env"], 99) or 1), 0, 1)
        panels.append(panel("audio: loudness (white) / onset strength (yellow)", [(audio["rms_t"], db, (230, 230, 230)), (audio["env_t"], env, (0, 200, 240))],
                            [(230, 230, 230), (0, 200, 240)], ymax=1.0, ticks=audio["onsets"]))
    axis = np.full((26, W, 3), 22, np.uint8)
    step = max(1, int(math.ceil(span / 12)))
    for sec in range(int(math.ceil(t0)), int(t1) + 1, step):
        cv2.line(axis, (X(sec), 0), (X(sec), 6), (150, 150, 150), 1)
        put_text(axis, f"{sec}s", (X(sec) - 10, 20), 0.36)
    return cv2.imwrite(str(path), np.vstack([strip] + panels + [axis]))


# ----------------------------------------------------------------------------- report

def ffmpeg_bgr_stream(video, start, dur, rate, w, h):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-an", "-vf", f"fps={rate:.5f},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 12)
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def render_view_video(video, S, shots, movers, audio, act, meta, out_path, start, end):
    """For a human: the video with cuts, camera motion, moving regions, tracked movers and an activity gauge drawn on."""
    vw = 540
    vh = even(meta["height"] * vw / meta["width"])
    gh, aw, ah, dt = 80, meta["aw"], meta["ah"], meta["dt"]
    n_steps = len(S["t"])
    t0, t1 = float(S["t"][0]), float(S["t"][-1]) + dt
    span = max(1e-6, t1 - t0)
    labels, shot_of = [], np.zeros(n_steps, int)
    for n, (a, b) in enumerate(shots):
        labels.append(camera_label(S, a, b, dt)[0])
        shot_of[a:b + 1] = n
    gauge = np.full((gh, vw, 3), 24, np.uint8)
    top = float(np.percentile(act, 99)) + 1e-9
    pts = [[int((S["t"][i] - t0) / span * (vw - 1)), int(gh - 6 - np.clip(act[i] / top, 0, 1) * (gh - 24))] for i in range(n_steps)]
    cv2.polylines(gauge, [np.array(pts, np.int32)], False, (80, 220, 80), 1, cv2.LINE_AA)
    for c in S["t"][S["cut"]]:
        x = int((c - t0) / span * (vw - 1))
        cv2.line(gauge, (x, 14), (x, gh - 2), (60, 60, 230), 1)
    if audio:
        for o in audio["onsets"]:
            if t0 <= o <= t1:
                x = int((o - t0) / span * (vw - 1))
                cv2.line(gauge, (x, gh - 10), (x, gh - 2), (0, 220, 255), 1)
    put_text(gauge, "motion energy   red = cuts   yellow = audio beats", (6, 12), 0.36)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{vw}x{vh + gh}", "-r", f"{1 / dt:.4f}", "-i", "-",
           "-ss", f"{t0:.3f}", "-t", f"{span:.3f}", "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "24", "-c:a", "aac", "-shortest", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    k = FLOW_W
    sc = vw / FLOW_W
    try:
        for i, frame in enumerate(ffmpeg_bgr_stream(video, t0, span, 1 / dt, vw, vh)):
            if i >= n_steps:
                break
            img = frame.copy()
            since_cut = 99
            for j in range(i, max(-1, i - 6), -1):
                if S["cut"][j]:
                    since_cut = i - j
                    break
            if since_cut < 6:
                cv2.rectangle(img, (0, 0), (vw - 1, vh - 1), (60, 60, 240), 6)
                put_text(img, "CUT", (vw - 70, 40), 0.9, (80, 80, 255), 2)
            for (bx, by, be, ba) in S["blobs"][i]:
                cv2.circle(img, (int(bx * sc), int(by * sc)), int(max(6, math.sqrt(ba / math.pi) * sc)), (0, 170, 255), 1, cv2.LINE_AA)
            for m in movers:
                if "tag" not in m or not (m["f0"] <= i <= m["f1"]):
                    continue
                idx = i - m["f0"]
                trail = []
                for q in range(max(0, idx - int(1.0 / dt)), idx + 1):
                    sp = S["M"][m["f0"] + q] @ np.array([m["pos"][q, 0], m["pos"][q, 1], 1.0])
                    trail.append((int(sp[0] / aw * vw), int(sp[1] / ah * vh)))
                for a_, b_ in zip(trail[:-1], trail[1:]):
                    cv2.line(img, a_, b_, (255, 160, 60), 2, cv2.LINE_AA)
                if trail:
                    cv2.circle(img, trail[-1], 5, (255, 255, 255), -1, cv2.LINE_AA)
                    put_text(img, m["tag"], (trail[-1][0] + 7, trail[-1][1] - 7), 0.6, (0, 255, 255), 2)
            lo, hi = max(0, i - 2), min(n_steps - 1, i + 2)
            tx, ty = -float(S["tx"][lo:hi + 1].mean()) * vw * 10, -float(S["ty"][lo:hi + 1].mean()) * vw * 10
            if math.hypot(tx, ty) > 6:
                c0 = (vw // 2, vh - 70)
                cv2.arrowedLine(img, c0, (int(c0[0] + tx), int(c0[1] + ty)), (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
                put_text(img, "camera", (c0[0] - 26, c0[1] + 22), 0.45)
            n = int(shot_of[i])
            put_text(img, f"watch motion  t={S['t'][i]:.1f}s  shot {n + 1}/{len(shots)}: {labels[n]}", (6, 16), 0.45)
            g = gauge.copy()
            cx = int((S["t"][i] - t0) / span * (vw - 1))
            cv2.line(g, (cx, 14), (cx, gh - 1), (255, 255, 255), 1)
            proc.stdin.write(np.vstack([img, g]).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    return Path(out_path).is_file()


def build_report(video, S, shots, movers, audio, meta, args, peaks, image_notes):
    dt, fps = meta["dt"], meta["fps"]
    t = S["t"]
    dur = float(t[-1] - t[0] + dt)
    mot = smooth(S["mag"], 3) / dt                  # frame widths per second
    obj = smooth(S["loc"], 3) / dt                   # strongest 1.5% of pixels: catches small fast objects
    act = np.maximum(mot / (np.percentile(mot, 95) + 1e-9), obj / (np.percentile(obj, 95) + 1e-9))
    cut_idx = [j for j in range(len(t)) if S["cut"][j]]
    cut_times = [float(t[j]) for j in cut_idx]
    lines = []
    A = lines.append
    A("# watch: motion report")
    A("")
    A(f"- **Source:** `{video}`")
    A(f"- **Analysed:** {fmt_t(t[0])} → {fmt_t(t[-1] + dt)} ({dur:.1f}s) at {1 / dt:.1f} fps "
      f"(source {fps:.1f} fps, {meta['width']}x{meta['height']}); {meta['steps']} frames compared pairwise")
    A(f"- **Cuts:** {len(cut_times)} hard cuts → {len(shots)} shots"
      + (f", avg shot {dur / len(shots):.2f}s" if shots else ""))
    A(f"- **Motion level:** whole-frame mean {mot.mean() * 100:.1f}% of frame width per second (peak {mot.max() * 100:.0f}%/s); "
      f"fastest-moving 1.5% of pixels, camera removed: mean {obj.mean() * 100:.1f}%/s (peak {obj.max() * 100:.0f}%/s)")
    A("")

    # hook & pacing
    A("## Hook and pacing")
    A("")
    hook_end = t[0] + 3.0
    hk = (t <= hook_end)
    rest = ~hk
    hook_mot = float(act[hk].mean()) if hk.any() else 0.0
    rest_mot = float(act[rest].mean()) if rest.any() else hook_mot
    hook_cuts = sum(1 for c in cut_times if c <= hook_end)
    A(f"- First 3 s: {hook_cuts} cut(s), visual activity {hook_mot:.2f} vs {rest_mot:.2f} for the rest (1.0 = typical busy moment) "
      f"({'busier start' if hook_mot > 1.15 * rest_mot else 'calmer start' if hook_mot < 0.85 * rest_mot else 'even energy'})")
    first_move = np.where(act > 0.3)[0]
    if len(first_move):
        A(f"- First clearly visible motion at {t[first_move[0]] - t[0]:.2f}s")
    lens = [float(t[b] - t[a] + dt) for a, b in shots]
    if len(lens) >= 3:
        half = len(lens) // 2
        a_, b_ = np.mean(lens[:half]), np.mean(lens[half:])
        trend = "accelerating (shots get shorter)" if b_ < 0.8 * a_ else "slowing (shots get longer)" if b_ > 1.25 * a_ else "steady"
        A(f"- Shot lengths: min {min(lens):.2f}s, median {np.median(lens):.2f}s, max {max(lens):.2f}s; first half avg {a_:.2f}s, second half {b_:.2f}s → {trend}")
    A("")

    # shots
    A("## Shots")
    A("")
    A("| # | time | len | camera | whole-frame %/s | object %/s |")
    A("|---|------|-----|--------|-----------|-----------|")
    shot_stats = []
    for n, (a, b) in enumerate(shots):
        lab, info = camera_label(S, a, b, dt)
        sl = slice(a + 1 if b > a else a, b + 1)
        m_, o_ = float(mot[sl].mean()) * 100 if b > a else 0.0, float(obj[sl].mean()) * 100 if b > a else 0.0
        shot_stats.append((n, a, b, lab, m_, o_, info))
        if len(shots) <= 40 or n < 20 or n >= len(shots) - 5:
            A(f"| {n + 1} | {t[a]:.2f}–{t[b] + dt:.2f}s | {t[b] - t[a] + dt:.2f}s | {lab} | {m_:.1f} | {o_:.1f} |")
    A("")

    # rhythm
    A("## Motion rhythm")
    A("")
    sec_edges = np.arange(t[0], t[-1] + 1.0, 1.0)

    def per_second(curve):
        return [float(curve[(t >= e) & (t < e + 1.0)].mean()) if ((t >= e) & (t < e + 1.0)).any() else 0.0 for e in sec_edges[:-1]]
    for name, curve in (("whole frame ", mot), ("object (local)", obj)):
        ps = per_second(curve)
        if not ps:
            continue
        for j in range(0, len(ps), 60):
            chunk = ps[j:j + 60]
            A(f"`{sparkline(chunk, 0, max(ps))}`  {name} {j}–{j + len(chunk)} s (one char = 1 s)")
    A("")
    if peaks:
        A("Strongest activity peaks (time: whole-frame / object motion in frame widths per second): "
          + ", ".join(f"{t[p]:.2f}s ({mot[p]:.2f} / {obj[p]:.2f})" for p in peaks[:8]))
        A("")
    cx, cy = float(np.average(S["cx"], weights=S["res"] + 1e-9)), float(np.average(S["cy"], weights=S["res"] + 1e-9))
    A(f"- Where the action is: centre of object motion averages at {where_words(cx, cy)} (x={cx:.2f}, y={cy:.2f}; 0,0 = top-left)")
    A("")

    # movers
    A("## Tracked movers")
    A("")
    if movers:
        A("Paths are camera-compensated. `A`, `B`, … match the labels in the tracks image. Speeds are frame widths per second.")
        A("")
        for n, m in enumerate(movers[:6]):
            e = m["easing"]
            tag = m["tag"]
            A(f"- **{tag}** shot {m['shot'] + 1}, {m['t0']:.2f}–{m['t1']:.2f}s: {where_words(*m['start_xy'])} → {where_words(*m['end_xy'])} "
              f"(net {m['net_dir']}, path {m['length_w']:.2f} frame widths), peak speed {m['peak_speed']:.2f} at {m['peak_t']:.2f}s, "
              f"{m['members']} feature tracks")
            lab = e.get("label", "?")
            if "rmse" in e:
                extra = f"; main move {e['duration']:.2f}s, fit error {e['rmse']:.3f}"
                if lab.startswith("ease-in (quad)") and m["net_dir"].startswith("down"):
                    extra += " — looks like gravity"
                A(f"  - motion curve: **{lab}**{extra}")
            else:
                A(f"  - motion curve: {lab}")
            A(f"  - speed profile: `{sparkline(m['speed'])}`")
            A(f"  - to follow it in detail: `track.py <video> --obj {tag}@{m['seed_t']:.2f}={m['seed_xy'][0]:.3f},{m['seed_xy'][1]:.3f}`")
    else:
        A("_No coherent moving objects found (static scene, or movement too fine/textureless for point tracking). "
          "Rely on the object-motion curve and the motion-history image._")
    A("")

    # audio
    A("## Audio sync")
    A("")
    if audio:
        tp = audio["tempo"]
        A(f"- {len(audio['onsets'])} audio onsets ({audio['onset_rate']:.1f}/s)"
          + (f"; estimated tempo ~{tp['bpm']:.0f} BPM (confidence {tp['confidence']:.2f}; low for speech-only audio)" if tp else ""))
        db = audio["rms_db"]
        A(f"- Loudness: mean {db.mean():.0f} dBFS, range {db.min():.0f}…{db.max():.0f}; first second {db[:4].mean():.0f} dBFS")
        for key, name in (("cut_sync", "cuts"), ("peak_sync", "motion peaks")):
            r = audio.get(key)
            if r:
                verdict = ("strongly beat-locked" if r["ratio"] and r["ratio"] > 2.0 else
                           "somewhat aligned" if r["ratio"] and r["ratio"] > 1.4 else "no clear alignment")
                A(f"- {name.capitalize()} vs onsets: {r['hit_rate'] * 100:.0f}% land within 80 ms of an onset "
                  f"(chance {r['chance'] * 100:.0f}%, ×{r['ratio']:.1f} over chance, n={r['n']}) → {verdict}; "
                  f"median offset {r['median_offset_ms']:+.0f} ms (positive = sound comes after the visual)")
    else:
        A("_No usable audio track._")
    A("")

    # images
    A("## Images to view")
    A("")
    A("Read these with the image tool. Each replaces dozens of stills:")
    A("")
    for p, note in image_notes:
        A(f"- `{p}` — {note}" if p else f"- _not rendered:_ {note}")
    A("")
    A("## How to read this")
    A("")
    A("- Motion-history and strobe images are camera-stabilised; a window is only used when the camera barely drifts. Tracks work with a moving camera. "
      "Dot spacing in the tracks image is the speed curve: dots that bunch up mean the object eases out (or in).")
    A("- Motion values are in *frame widths per second* so they compare across resolutions; ×100 for % of the frame width per second.")
    A("- *Camera* rows describe what the camera does (content moving right = camera panning left). *Object* motion has the camera model subtracted.")
    A("- Hard cuts only: dissolves and fades are not counted as cuts. Easing labels are fits of the main move to standard curves; treat RMSE > 0.08 as a weak match.")
    A("- Beat-sync uses a shuffle test: the ratio says how much better than random placement the cuts or peaks hit audio onsets.")
    return "\n".join(lines), shot_stats


def pick_peaks(mot, dt, S):
    clean = mot.copy()
    clean[S["cut"]] = np.median(clean)
    prom = max(0.25 * float(np.percentile(clean, 95) - np.percentile(clean, 5)), 1e-6)
    pk, props = signal.find_peaks(clean, prominence=prom, distance=max(1, int(0.3 / dt)))
    order = np.argsort(-props["prominences"])
    return [int(pk[j]) for j in order[:12]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Motion analysis for /watch --motion")
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--max-shots", type=int, default=3, help="Shots that get detailed images")
    ap.add_argument("--fps-cap", type=float, default=30.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-view", action="store_true", help="Skip the annotated motion_view.mp4")
    args = ap.parse_args()
    log = lambda m: print(m, file=sys.stderr, flush=True)
    video = Path(args.video).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    S, tracks, meta = pass1(video, args.start, args.end, args.fps_cap, log)
    dt = meta["dt"]
    shots = build_shots(S, dt)
    mot = smooth(S["mag"], 3) / dt
    obj = smooth(S["loc"], 3) / dt
    act = np.maximum(mot / (np.percentile(mot, 95) + 1e-9), obj / (np.percentile(obj, 95) + 1e-9))
    peaks = pick_peaks(act, dt, S)
    movers = finish_movers(cluster_movers(tracks, S, meta) + link_blobs(S, meta), S, meta)
    dur = float(S["t"][-1] - S["t"][0] + dt)

    audio = None
    if not args.no_audio:
        cut_times = [float(x) for x in S["t"][S["cut"]]]
        audio = analyse_audio(video, args.start, args.end, cut_times, [float(S["t"][p]) for p in peaks], dur, log)

    log("[motion] rendering images…")
    image_notes = []
    skipped = []
    ov = out / "overview.png"
    if render_overview(video, S, shots, peaks, audio, meta, ov):
        image_notes.append((ov, "thumbnail per shot + motion / object-motion / camera / audio curves on one time axis (red = cuts, yellow = audio onsets)"))

    # detailed images for the first shot (the hook) and the most active shots
    activity = []
    for n, (a, b) in enumerate(shots):
        if b - a >= 6:
            activity.append((float(S["loc"][a + 1:b + 1].mean()) * math.sqrt(b - a), n))
    chosen = {0} if shots and shots[0][1] - shots[0][0] >= 6 else set()
    for _, n in sorted(activity, reverse=True):
        if len(chosen) >= args.max_shots:
            break
        chosen.add(n)
    for n in sorted(chosen):
        a, b = shots[n]
        win = stable_window(S, a, b, meta)
        sf = window_frames(video, S, win, meta)
        t_a, t_b = S["t"][win["j0"]], S["t"][win["j1"]] + dt
        how = "whole shot" if win["full"] else f"best {t_b - t_a:.1f}s window, camera moves"
        label = f"shot {n + 1} {t_a:.1f}-{t_b:.1f}s"
        if win["stable"]:
            p = out / f"shot{n + 1:02d}_motion_history.png"
            if render_history(sf, S, p, label):
                image_notes.append((p, f"{label} [{how}]: everything that moved, painted blue (early) → red (late) over the empty background"))
            p = out / f"shot{n + 1:02d}_strobe.png"
            if render_strobe(sf, S, p, label):
                image_notes.append((p, f"{label} [{how}]: the moving subject stamped at 7 moments (chronophotography), outline colour = time"))
        else:
            skipped.append(f"shot {n + 1} ({t_a:.1f}-{t_b:.1f}s): camera moves too much for a clean motion-history / strobe overlay")
        p = out / f"shot{n + 1:02d}_tracks.png"
        if render_tracks(sf, S, win, movers, meta, p, label):
            image_notes.append((p, f"{label} [{how}]: tracked object paths, a dot every 0.1 s (tight dots = slow, wide = fast), letters match the movers list"))
    for note in skipped:
        image_notes.append((None, note))

    if not args.no_view:
        log("[motion] rendering the annotated video for humans…")
        try:
            if render_view_video(video, S, shots, movers, audio, act, meta, out / "motion_view.mp4", args.start, args.end):
                image_notes.insert(0, (out / "motion_view.mp4", "FOR THE HUMAN, send this to the user: the video with cuts (red frame), camera arrow, moving regions, tracked movers A, B… and an activity gauge drawn on. The model cannot watch it."))
        except (OSError, ValueError) as exc:
            log(f"[motion] annotated video skipped: {exc}")
    report, shot_stats = build_report(video, S, shots, movers, audio, meta, args, peaks, image_notes)
    (out / "report.md").write_text(report, encoding="utf-8")
    data = dict(meta=meta, t=S["t"].round(4).tolist(), motion=(mot).round(5).tolist(), object_motion=(smooth(S["loc"], 3) / dt).round(5).tolist(),
                cuts=[float(x) for x in S["t"][S["cut"]]], peaks=[float(S["t"][p]) for p in peaks],
                cam=dict(tx=S["tx"].round(5).tolist(), ty=S["ty"].round(5).tolist(), zoom=S["zoom"].round(5).tolist(), rot=S["rot"].round(5).tolist()),
                shots=[dict(index=n + 1, start=float(S["t"][a]), end=float(S["t"][b] + dt), camera=lab, motion=m_, object=o_, camera_info=info)
                       for n, a, b, lab, m_, o_, info in shot_stats],
                movers=[dict(shot=m["shot"] + 1, t0=m["t0"], t1=m["t1"], net_dir=m["net_dir"], length_w=m["length_w"], peak_speed=m["peak_speed"],
                             easing={k: v for k, v in m["easing"].items()}, path=(m["pos"] / meta["aw"]).round(4).tolist()) for m in movers[:8]],
                audio=None if not audio else dict(onsets=audio["onsets"].round(3).tolist(), tempo=audio["tempo"], cut_sync=audio["cut_sync"], peak_sync=audio["peak_sync"]))
    (out / "report.json").write_text(json.dumps(data, default=float), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

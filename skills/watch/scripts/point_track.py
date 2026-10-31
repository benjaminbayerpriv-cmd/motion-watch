#!/usr/bin/env python3
"""CoTracker3 point tracking for /watch. Runs inside the managed tracker venv (PyTorch); `motion_track.py` launches it.

Follows the points you give (normalised 0..1) forward and backward from a seed time and writes their
positions and visibility per frame as JSON. Nothing is detected: the model tracks exactly the clicked spot.

Model: CoTracker3 online (Meta, CC BY-NC 4.0, non-commercial). Code and weights are pinned (see motion.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

REF = os.environ.get("WATCH_COTRACKER_REF", "facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ffmpeg_frames(video, t_a, t_b, rate, w, h):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_a):.3f}", "-i", str(video), "-t", f"{max(0.05, t_b - t_a):.3f}",
           "-an", "-vf", f"fps={rate:.4f},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def forward(video, t_a, t_b, rate, w, h):
    """Stream (time, frame) forward in time, a few seconds at a time to bound memory."""
    t, chunk = t_a, 6.0
    first = True
    while t < t_b - 0.5 / rate:
        end = min(t_b, t + chunk)
        frames = ffmpeg_frames(video, t, end + 0.02, rate, w, h)
        for k in range(len(frames)):
            tt = t + k / rate
            if tt <= end - (0.5 / rate if end < t_b else -1e-6):
                yield tt, frames[k]
        t = end
        first = False


def backward(video, t_a, t_b, rate, w, h, chunk=4.0):
    """Stream (time, frame) from t_b back to t_a."""
    end = t_b
    while end > t_a + 0.5 / rate:
        start = max(t_a, end - chunk)
        frames = ffmpeg_frames(video, start, end + 0.02, rate, w, h)
        for k in range(len(frames) - 1, -1, -1):
            tt = start + k / rate
            if tt <= end + 1e-6:
                yield tt, frames[k]
        end = start - 1.0 / rate


def track(model, dev, frames, pts_px):
    """frames: iterator of (t, RGB uint8). Query points sit on the first frame. Returns times, tracks (T,N,2), visible (T,N)."""
    step = model.step
    q = torch.tensor([[[0.0, x, y] for x, y in pts_px]], dtype=torch.float32, device=dev)
    window, times, first, out = [], [], True, (None, None)
    total = 0

    def chunk(frames_):
        arr = np.stack(frames_)
        return torch.from_numpy(arr).to(dev).float().permute(0, 3, 1, 2)[None]

    for i, (t, fr) in enumerate(frames):
        if i % step == 0 and i != 0:
            out = model(chunk(window[-step * 2:]), is_first_step=first, queries=q if first else None, grid_size=0)
            first = False
        window.append(fr)
        times.append(t)
        window = window[-step * 3:]
        total = i + 1
    if total < 3:
        raise SystemExit("Too few frames to track.")
    while len(window) < step + 2:                      # very short clips: repeat the last frame
        window.append(window[-1])
    tail = window[-((total - 1) % step) - step - 1:] if total - 1 >= step else window
    if first:                                          # the loop never reached a first step
        model(chunk(tail), is_first_step=True, queries=q, grid_size=0)
    out = model(chunk(tail), is_first_step=False, grid_size=0)
    tracks, vis = out
    tr = tracks[0].float().cpu().numpy()
    vi = vis[0].float().cpu().numpy() > 0.5
    n = min(len(times), len(tr))
    return np.array(times[:n]), tr[:n], vi[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-t", type=float, required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--points", required=True, help="x,y;x,y normalised 0..1")
    args = ap.parse_args()
    pts = [[float(v) for v in p.split(",")] for p in args.points.split(";")]
    pts_px = [[x * (args.width - 1), y * (args.height - 1)] for x, y in pts]
    dev = pick_device()
    log(f"[cotracker] loading CoTracker3 on {dev}…")
    model = torch.hub.load(REF, "cotracker3_online", trust_repo=True).to(dev).eval()
    result = {"rate": args.rate, "device": dev, "points": [{"fwd": None, "bwd": None} for _ in pts]}
    t0 = time.time()
    with torch.no_grad():
        if args.end - args.seed_t > 1.0 / args.rate:
            log(f"[cotracker] forward {args.seed_t:.1f}s → {args.end:.1f}s")
            times, tr, vi = track(model, dev, forward(args.video, args.seed_t, args.end, args.rate, args.width, args.height), pts_px)
            for j in range(len(pts)):
                result["points"][j]["fwd"] = {"t": times.tolist(), "x": tr[:, j, 0].tolist(), "y": tr[:, j, 1].tolist(), "vis": vi[:, j].tolist()}
        if args.seed_t - args.start > 1.0 / args.rate:
            log(f"[cotracker] backward {args.seed_t:.1f}s → {args.start:.1f}s")
            times, tr, vi = track(model, dev, backward(args.video, args.start, args.seed_t, args.rate, args.width, args.height), pts_px)
            for j in range(len(pts)):
                result["points"][j]["bwd"] = {"t": times.tolist(), "x": tr[:, j, 0].tolist(), "y": tr[:, j, 1].tolist(), "vis": vi[:, j].tolist()}
    result["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

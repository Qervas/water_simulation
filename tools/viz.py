#!/usr/bin/env python3
"""
viz.py — quick visualizer for water_sim recorded frames.

Reads frame_NNNN.bin files (uint32 count + Nx vec3<float32>) from a directory
and shows a 3D scatter animation. Optionally writes an mp4 via ffmpeg-backed
matplotlib.animation.

Usage:
    python tools/viz.py out/falling_block            # interactive window
    python tools/viz.py out/falling_block --mp4 viz.mp4   # save mp4
    python tools/viz.py out/falling_block --fps 30 --stride 1
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib

# Pick the backend BEFORE importing pyplot. If --mp4 was passed, force Agg
# (headless). Otherwise try interactive backends in order of preference.
_mp4_mode = "--mp4" in sys.argv
if _mp4_mode:
    matplotlib.use("Agg")
else:
    for backend in ("TkAgg", "QtAgg", "WebAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection


def load_frame(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    pts = np.frombuffer(data, dtype=np.float32, count=3 * n, offset=4).reshape(n, 3)
    return pts


def discover_frames(d):
    files = sorted(Path(d).glob("frame_*.bin"))
    if not files:
        sys.exit(f"viz: no frame_*.bin files in {d}")
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frame_dir", help="directory containing frame_NNNN.bin files")
    p.add_argument("--mp4", default=None, help="write mp4 to this path instead of showing live")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1, help="show every Nth frame")
    p.add_argument("--size", type=float, default=20.0, help="point size")
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--bbox", default="0,0,0,1,1,1",
                   help="domain bbox: xmin,ymin,zmin,xmax,ymax,zmax")
    args = p.parse_args()

    files = discover_frames(args.frame_dir)[::args.stride]
    print(f"viz: {len(files)} frames, first={files[0].name}, last={files[-1].name}")

    bbox = list(map(float, args.bbox.split(",")))
    assert len(bbox) == 6

    # Pre-load all frames (small enough — < 100k particles per frame typical).
    all_pts = [load_frame(f) for f in files]
    n_max = max(p.shape[0] for p in all_pts)
    print(f"viz: {all_pts[0].shape[0]} particles per frame")

    fig = plt.figure(figsize=(8, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(bbox[0], bbox[3])
    ax.set_ylim(bbox[2], bbox[5])     # matplotlib y = world z (depth)
    ax.set_zlim(bbox[1], bbox[4])     # matplotlib z = world y (up)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.view_init(elev=args.elev, azim=args.azim)

    # Domain wireframe
    xs = [bbox[0], bbox[3]]
    ys = [bbox[1], bbox[4]]
    zs = [bbox[2], bbox[5]]
    for x in xs:
        for y in ys:
            ax.plot([x, x], [zs[0], zs[1]], [y, y], color="gray", alpha=0.25, linewidth=0.5)
        for z in zs:
            ax.plot([x, x], [z, z], [ys[0], ys[1]], color="gray", alpha=0.25, linewidth=0.5)
    for y in ys:
        for z in zs:
            ax.plot([xs[0], xs[1]], [z, z], [y, y], color="gray", alpha=0.25, linewidth=0.5)

    initial = all_pts[0]
    # Color by initial Y position (so we can see vertical mixing)
    colors = initial[:, 1].copy()
    scat = ax.scatter(initial[:, 0], initial[:, 2], initial[:, 1],
                      c=colors, s=args.size, cmap="viridis",
                      vmin=bbox[1], vmax=bbox[4],
                      edgecolors="none", alpha=0.85)
    title = ax.set_title("frame 0")

    def update(i):
        pts = all_pts[i]
        scat._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])
        title.set_text(f"frame {i*args.stride}/{(len(files)-1)*args.stride}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(all_pts),
                         interval=1000.0 / args.fps, blit=False)

    if args.mp4:
        print(f"viz: writing {args.mp4} ...")
        writer = FFMpegWriter(fps=args.fps, bitrate=4000)
        anim.save(args.mp4, writer=writer)
        print(f"viz: wrote {args.mp4} ({Path(args.mp4).stat().st_size / 1024:.1f} KB)")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

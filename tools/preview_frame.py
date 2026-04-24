#!/usr/bin/env python3
"""Render a single frame of a recorded simulation as a PNG."""
import argparse
import struct
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_frame(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    return np.frombuffer(data, dtype=np.float32, count=3 * n, offset=4).reshape(n, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frame_path")
    p.add_argument("--out", required=True)
    p.add_argument("--bbox", default="0,0,0,1,1,1")
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--size", type=float, default=20.0)
    args = p.parse_args()

    pts = load_frame(Path(args.frame_path))
    bbox = list(map(float, args.bbox.split(",")))

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(bbox[0], bbox[3])
    ax.set_ylim(bbox[2], bbox[5])
    ax.set_zlim(bbox[1], bbox[4])
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               c=pts[:, 1], s=args.size, cmap="viridis",
               vmin=bbox[1], vmax=bbox[4],
               edgecolors="none", alpha=0.85)
    ax.set_title(f"{Path(args.frame_path).name} — n={len(pts)}")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
particles_to_mesh.py — convert recorded SPH particle binaries into smooth
fluid surface meshes via Open3D Poisson reconstruction.

This is an interim approach for surface reconstruction while a proper
anisotropic-kernel CUDA implementation (Yu & Turk 2013, Phase 4) is built.

Pipeline per frame:
  1. Read binary (uint32 count + N * vec3<float32>)
  2. Build Open3D point cloud
  3. Estimate per-particle normals (PCA on KNN neighborhood)
  4. Orient normals consistently (MST traversal)
  5. Poisson surface reconstruction
  6. Crop low-density mesh regions (Poisson over-extrapolates outside the
     point cloud — those regions get zero support and produce phantom
     surface; crop by density quantile)
  7. Optional Laplacian smoothing
  8. Write PLY

Usage:
    python tools/particles_to_mesh.py \\
        --seq out/dam_break_hires \\
        --out-dir out/dam_break_hires_mesh \\
        --radius 0.008 --depth 8 --stride 2

Or single-frame test:
    python tools/particles_to_mesh.py \\
        --frame out/dam_break/frame_0030.bin \\
        --out out/test_mesh.ply --radius 0.012
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage
from scipy.spatial import cKDTree
from skimage import measure


def load_frame(path):
    """Read binary frame; returns Nx3 numpy array in BLENDER Z-up coords.

    Sim is Y-up: (sim_x, sim_y_up, sim_z_depth).
    Blender is Z-up: (bl_x, bl_y_forward, bl_z_up).
    Convert by swapping Y and Z so the PLY mesh drops directly into the
    Blender scene without any further transform.
    """
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    pts = np.frombuffer(data, dtype=np.float32, count=3 * n,
                          offset=4).reshape(n, 3).astype(np.float64)
    # Swap columns 1 and 2: (x, y, z) -> (x, z, y)
    return pts[:, [0, 2, 1]]


def smooth_positions_knn(pts, k=15, iters=2):
    """Zhu-Bridson 2005 style: iteratively replace each particle position
    with the mean of its K nearest neighbors. Reduces particle clustering
    and produces dramatically smoother surfaces after density-splat + MC.
    Vectorized via cKDTree (fast even for 100K+ particles).
    """
    smoothed = pts.copy()
    for _ in range(iters):
        tree = cKDTree(smoothed)
        _, idx = tree.query(smoothed, k=k)
        # Average each particle with its k-NN neighbors
        smoothed = smoothed[idx].mean(axis=1)
    return smoothed


def reconstruct_splat(pts, particle_radius,
                       grid_voxel_factor=0.5,
                       smooth_sigma_voxels=2.5,
                       iso_factor=0.18,
                       smooth_iters_pre=2,
                       smooth_iters_post=2,
                       k_neighbors=15):
    """Splat-and-smooth surface reconstruction (Zhu-Bridson + Gaussian + MC).

    Approximates Yu-Turk anisotropic reconstruction at far lower cost.
    Pipeline:
      1. Smooth particle positions via K-NN averaging (reduces clustering).
      2. Splat smoothed particles to a 3D density grid.
      3. Gaussian-blur the density grid.
      4. Marching cubes at iso-level = iso_factor * max_density.
      5. Optional Laplacian smoothing of the resulting mesh.

    Result: surfaces dramatically smoother than Poisson when particles are
    densely packed (e.g. settled fluid). Bumps come out as wave-scale
    rather than particle-scale.
    """
    # 1. Smooth positions
    if smooth_iters_pre > 0:
        pts_s = smooth_positions_knn(pts, k=k_neighbors, iters=smooth_iters_pre)
    else:
        pts_s = pts

    # 2. Determine grid bounds and resolution
    pad = particle_radius * 4.0
    lo = pts_s.min(0) - pad
    hi = pts_s.max(0) + pad
    extent = hi - lo
    voxel = particle_radius * grid_voxel_factor
    res = np.maximum(np.ceil(extent / voxel).astype(int), 8)

    # 3. Splat — vectorized scatter into 3D grid
    grid = np.zeros(tuple(res), dtype=np.float32)
    grid_coords = ((pts_s - lo) / extent * res).astype(int)
    grid_coords = np.clip(grid_coords, 0, res - 1)
    np.add.at(grid, (grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]), 1.0)

    # 4. Gaussian blur (sigma in voxels)
    grid = scipy.ndimage.gaussian_filter(grid, sigma=smooth_sigma_voxels)

    max_d = grid.max()
    if max_d <= 0:
        return o3d.geometry.TriangleMesh()
    iso = max_d * iso_factor

    # 5. Marching cubes
    try:
        verts, faces, _normals, _ = measure.marching_cubes(
            grid, level=iso, spacing=tuple(extent / res))
    except (ValueError, RuntimeError):
        return o3d.geometry.TriangleMesh()
    verts = verts + lo

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if smooth_iters_post > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iters_post)
    mesh.compute_vertex_normals()
    return mesh


def reconstruct(pts, particle_radius,
                depth=8,
                density_quantile=0.05,
                smooth_iters=2):
    """Particles → smooth water surface mesh."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Normal estimation: hybrid radius + KNN. Radius ≈ 4*particle_radius
    # gives enough neighbors to PCA-fit a local plane.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=particle_radius * 5.0, max_nn=30))
    # MST-based orientation makes adjacent normals point the same way.
    # k=10 means each normal is consistent with its 10 nearest neighbors.
    try:
        pcd.orient_normals_consistent_tangent_plane(k=10)
    except Exception as e:
        # Fallback: orient toward centroid (approximate but won't crash).
        sys.stderr.write(f"  warn: tangent-plane orientation failed ({e}), "
                         "using centroid heuristic\n")
        centroid = pts.mean(axis=0)
        normals = pts - centroid
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Poisson reconstruction. depth=8 → ~256³ resolution.
    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=1.05, linear_fit=False))

    # Crop low-density vertices. Poisson smoothly extrapolates beyond the
    # point cloud's actual extent; the extrapolation has near-zero
    # density support and produces phantom surface around the fluid.
    densities = np.asarray(densities)
    if len(densities) > 0:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    # Optional Laplacian smoothing — softens any remaining bumpiness
    # without losing surface features. Number of iterations is a knob.
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iters)
        mesh.compute_vertex_normals()

    return mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", help="single binary frame path")
    p.add_argument("--seq", help="directory of frame_NNNN.bin")
    p.add_argument("--out", help="single output .ply (with --frame)")
    p.add_argument("--out-dir", help="output directory (with --seq)")
    p.add_argument("--radius", type=float, default=0.012,
                   help="particle radius from sim")
    p.add_argument("--method", choices=("poisson", "splat"), default="splat",
                   help="surface reconstruction method")
    # Poisson params
    p.add_argument("--depth", type=int, default=8,
                   help="[poisson] octree depth (8=256^3, 9=512^3)")
    p.add_argument("--density-quantile", type=float, default=0.05,
                   help="[poisson] crop vertices with density below this quantile")
    # Splat params
    p.add_argument("--grid-voxel-factor", type=float, default=0.5,
                   help="[splat] density-grid voxel size = factor * particle_radius")
    p.add_argument("--smooth-sigma", type=float, default=2.5,
                   help="[splat] Gaussian blur sigma (in voxels)")
    p.add_argument("--iso-factor", type=float, default=0.18,
                   help="[splat] iso-level = factor * max_density")
    p.add_argument("--smooth-pre", type=int, default=2,
                   help="[splat] particle-position smoothing iterations (Zhu-Bridson)")
    p.add_argument("--k-neighbors", type=int, default=15,
                   help="[splat] K-NN count for position smoothing")
    # Common
    p.add_argument("--smooth", type=int, default=2,
                   help="post-MC Laplacian mesh smoothing iterations (0 disables)")
    p.add_argument("--stride", type=int, default=1,
                   help="process every Nth frame")
    args = p.parse_args()

    def process_one(frame_path, out_path):
        pts = load_frame(Path(frame_path))
        t0 = time.time()
        if args.method == "poisson":
            mesh = reconstruct(pts, args.radius, args.depth,
                                args.density_quantile, args.smooth)
        else:  # splat
            mesh = reconstruct_splat(
                pts, args.radius,
                grid_voxel_factor=args.grid_voxel_factor,
                smooth_sigma_voxels=args.smooth_sigma,
                iso_factor=args.iso_factor,
                smooth_iters_pre=args.smooth_pre,
                smooth_iters_post=args.smooth,
                k_neighbors=args.k_neighbors)
        t1 = time.time()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return len(pts), len(mesh.vertices), len(mesh.triangles), t1 - t0

    if args.frame:
        if not args.out:
            sys.exit("--out required with --frame")
        n_pts, n_v, n_t, dt = process_one(args.frame, args.out)
        print(f"  {args.frame}: {n_pts} particles → {n_v} verts, "
              f"{n_t} tris ({dt:.1f}s)")
    elif args.seq:
        if not args.out_dir:
            sys.exit("--out-dir required with --seq")
        files = sorted(Path(args.seq).glob("frame_*.bin"))[::args.stride]
        for i, f in enumerate(files):
            out = Path(args.out_dir) / f"{f.stem}.ply"
            n_pts, n_v, n_t, dt = process_one(str(f), str(out))
            print(f"[{i+1}/{len(files)}] {f.name} → {out.name}: "
                  f"{n_pts} particles → {n_v} verts, {n_t} tris ({dt:.1f}s)")
    else:
        sys.exit("must pass --frame or --seq")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
blender_render.py — render simulation frames as photoreal water via Cycles.

Coordinate convention:
    Simulation world is Y-up (sim X=right, sim Y=up, sim Z=depth).
    Blender world is Z-up (Blender X=right, Blender Y=forward, Blender Z=up).
    Conversion: (sim_x, sim_y, sim_z) -> (sim_x, sim_z, sim_y).

Run with:
    blender --background --python tools/blender_render.py -- \\
        --frame out/dam_break/frame_0000.bin \\
        --out   out/cycles_hero.png --spp 64

Useful debug flag:
    --no-water           render the room only (no fluid)
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import bpy
import mathutils  # type: ignore


# ---------- coordinate conversion ----------

def sim_to_blender(p):
    """(sim_x, sim_y_up, sim_z_depth) -> (bl_x, bl_y_forward, bl_z_up)."""
    return (p[0], p[2], p[1])


def load_frame(path):
    """Read binary frame; returns positions already in Blender Z-up coords."""
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    pts = []
    for i in range(n):
        x, y, z = struct.unpack_from("<fff", data, 4 + 12 * i)
        pts.append(sim_to_blender((x, y, z)))
    return pts


# ---------- scene setup helpers ----------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(spp: int, res_x: int, res_y: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # Bleeding-edge GPUs (RTX 5070 sm_120) often don't match Blender's bundled
    # CUDA kernels yet. Use CPU — bulletproof. ~12s per frame at 540p/64spp.
    scene.cycles.device = "CPU"
    scene.cycles.samples = spp
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass


def make_principled(name, rgb, roughness, transmission=0.0, ior=1.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    for k in ("Transmission Weight", "Transmission"):
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = transmission
            break
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat


def make_water_material():
    return make_principled("Water", (0.85, 0.92, 1.0), roughness=0.0,
                            transmission=1.0, ior=1.333)


def setup_room(domain_size=1.0):
    """Build the room: floor at z=0, two walls at -X and -Y, all in Z-up."""
    # Floor — large wood plane in XY at z=0. Default plane is in XY with normal +Z. ✓
    bpy.ops.mesh.primitive_plane_add(
        size=domain_size * 8.0,
        location=(domain_size / 2, domain_size / 2, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(make_principled(
        "Floor", (0.20, 0.13, 0.08), roughness=0.55))

    # Back wall at -Y, facing +Y (toward camera). Default plane normal is +Z.
    # To get a vertical wall at y = -X, with normal in +Y: rotate +90° around X.
    bpy.ops.mesh.primitive_plane_add(
        size=domain_size * 8.0,
        location=(domain_size / 2, -domain_size * 1.5, domain_size * 1.0),
        rotation=(math.radians(90), 0, 0))
    bw = bpy.context.active_object
    bw.name = "BackWall"
    bw.data.materials.append(make_principled(
        "BackWall", (0.10, 0.25, 0.45), roughness=0.65))

    # Side wall at -X, facing +X. Rotate +90° around Y to put plane in YZ.
    bpy.ops.mesh.primitive_plane_add(
        size=domain_size * 8.0,
        location=(-domain_size * 1.5, domain_size / 2, domain_size * 1.0),
        rotation=(0, math.radians(90), 0))
    sw = bpy.context.active_object
    sw.name = "SideWall"
    sw.data.materials.append(make_principled(
        "SideWall", (0.45, 0.20, 0.15), roughness=0.65))


def _aim_at(obj, target):
    """Add a Track-To constraint so the object's -Z axis points at target."""
    bpy.ops.object.empty_add(location=target)
    aim = bpy.context.active_object
    aim.name = obj.name + ".target"
    c = obj.constraints.new(type="TRACK_TO")
    c.target = aim
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"


def setup_lighting(domain_size=1.0):
    """Three-point area lighting; each light auto-aims at scene center."""
    target = (domain_size / 2, domain_size / 2, domain_size * 0.3)

    # Key (warm), high above and slightly +X +Y
    bpy.ops.object.light_add(
        type="AREA",
        location=(domain_size * 1.8, domain_size * 1.5, domain_size * 2.5))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 200
    key.data.size = 2.0
    key.data.color = (1.0, 0.95, 0.85)
    _aim_at(key, target)

    # Fill (cool), opposite side, weaker
    bpy.ops.object.light_add(
        type="AREA",
        location=(-domain_size * 0.5, -domain_size * 0.5, domain_size * 1.8))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 100
    fill.data.size = 2.0
    fill.data.color = (0.85, 0.90, 1.0)
    _aim_at(fill, target)

    # Rim/back (warm, kicks edges)
    bpy.ops.object.light_add(
        type="AREA",
        location=(domain_size / 2, -domain_size * 1.5, domain_size * 0.8))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 80
    rim.data.size = 1.0
    rim.data.color = (1.0, 0.92, 0.80)
    _aim_at(rim, target)

    # Gradient sky world background — gentle ambient
    bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "EASING"
    map_ = nt.nodes.new("ShaderNodeMapping")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    cramp = nt.nodes.new("ShaderNodeValToRGB")
    cramp.color_ramp.elements[0].color = (0.10, 0.13, 0.18, 1.0)
    cramp.color_ramp.elements[1].color = (0.55, 0.65, 0.80, 1.0)
    nt.links.new(coord.outputs["Generated"], map_.inputs["Vector"])
    nt.links.new(map_.outputs["Vector"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Fac"], cramp.inputs["Fac"])
    nt.links.new(cramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.4
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_camera(domain_size=1.0):
    """Camera positioned at +X+Y+Z corner, tracking the scene center."""
    target = (domain_size / 2, domain_size / 2, domain_size * 0.25)
    cam_pos = (domain_size * 2.4, domain_size * 2.4, domain_size * 1.4)

    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.data.lens = 35
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = math.sqrt(
        sum((c - t) ** 2 for c, t in zip(cam_pos, target)))
    cam.data.dof.aperture_fstop = 8.0

    bpy.ops.object.empty_add(location=target)
    aim = bpy.context.active_object
    aim.name = "CamTarget"
    track = cam.constraints.new(type="TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


def make_metaball_cluster(pts, radius, water_mat):
    """Metaballs auto-merge into a fluid surface. Pts are already Z-up Blender."""
    mball = bpy.data.metaballs.new("FluidMeta")
    mball.resolution = radius * 0.5
    mball.render_resolution = radius * 0.3
    mball.threshold = 0.25
    obj = bpy.data.objects.new("Fluid", mball)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(water_mat)
    for p in pts:
        e = mball.elements.new(type="BALL")
        e.co = p
        e.radius = radius * 4.0
    return obj


# ---------- top-level ----------

def render_one(out_path: str, frame_path: str = None,
                spp=64, res=(1280, 720), radius=0.012, domain_size=1.0,
                no_water=False):
    reset_scene()
    setup_render(spp, res[0], res[1])
    setup_room(domain_size)
    setup_lighting(domain_size)
    setup_camera(domain_size)

    if not no_water:
        if frame_path is None:
            sys.exit("frame_path required when no_water=False")
        pts = load_frame(frame_path)
        print(f"  loaded {len(pts)} particles from {frame_path}")
        make_metaball_cluster(pts, radius, make_water_material())
    else:
        print("  --no-water: rendering empty room")

    bpy.context.scene.render.filepath = str(Path(out_path).resolve())
    bpy.ops.render.render(write_still=True)
    print(f"  wrote {out_path}")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--frame", help="single binary frame")
    p.add_argument("--seq", help="directory of frame_NNNN.bin (batch)")
    p.add_argument("--out", help="single output PNG")
    p.add_argument("--out-dir", help="batch output directory")
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--res", nargs=2, type=int, default=[1280, 720])
    p.add_argument("--radius", type=float, default=0.012)
    p.add_argument("--domain", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-water", action="store_true",
                   help="render the room only, no fluid")
    args = p.parse_args(argv)

    if args.no_water:
        if not args.out:
            sys.exit("--out required with --no-water")
        render_one(args.out, frame_path=None, spp=args.spp,
                    res=tuple(args.res), radius=args.radius,
                    domain_size=args.domain, no_water=True)
        return

    if args.frame:
        if not args.out:
            sys.exit("--out required with --frame")
        render_one(args.out, frame_path=args.frame, spp=args.spp,
                    res=tuple(args.res), radius=args.radius,
                    domain_size=args.domain)
    elif args.seq:
        if not args.out_dir:
            sys.exit("--out-dir required with --seq")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        files = sorted(Path(args.seq).glob("frame_*.bin"))[::args.stride]
        for i, f in enumerate(files):
            out = Path(args.out_dir) / f"render_{f.stem.split('_')[1]}.png"
            print(f"[{i+1}/{len(files)}] {f.name} -> {out.name}")
            render_one(str(out), frame_path=str(f), spp=args.spp,
                        res=tuple(args.res), radius=args.radius,
                        domain_size=args.domain)
    else:
        sys.exit("must pass --frame, --seq, or --no-water")


if __name__ == "__main__":
    main()

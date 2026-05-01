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


def setup_render(spp: int, res_x: int, res_y: int, caustics: bool = True):
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

    if caustics:
        # Cycles defaults clamp/disable refractive caustics for noise reduction.
        # Re-enable so light can bend through the water column onto the floor.
        # Cost: noisier render, must be paid back with higher SPP + denoising.
        for attr, val in (
            ("caustics_reflective", True),
            ("caustics_refractive", True),
            ("transmission_bounces", 24),
            ("max_bounces", 16),
            ("blur_glossy", 0.5),  # reduces firefly noise on caustic spots
        ):
            try:
                setattr(scene.cycles, attr, val)
            except Exception:
                pass
        # sample_clamp_indirect=0 disables clamping (preserves caustic peaks).
        # Setting too low (e.g. 1.0) kills caustics; 10.0 is a safe firefly cap.
        try:
            scene.cycles.sample_clamp_indirect = 10.0
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
    """Water with slight aqua tint (real water absorbs red wavelengths at
    depth — gives the characteristic blue-green look that distinguishes
    water from clear glass at the same scale)."""
    return make_principled("Water", (0.78, 0.92, 0.96), roughness=0.0,
                            transmission=1.0, ior=1.333)


def make_glass_material():
    """Tank glass — colorless, IOR 1.50."""
    return make_principled("TankGlass", (0.97, 0.98, 1.0), roughness=0.0,
                            transmission=1.0, ior=1.5)


def _box(name, lo, hi, material):
    """Add an axis-aligned box from `lo` (corner) to `hi` (corner), Blender Z-up.
    Default cube primitive is centered with side=2, so we use size=2 then scale
    by half-extents and translate to center.
    """
    cx, cy, cz = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2
    sx, sy, sz = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.scale = (sx, sy, sz)
    obj.name = name
    obj.data.materials.append(material)
    return obj


def setup_room(domain_size=1.0):
    """Counter + glass tank around the sim domain + back wall.

    Sim domain is [0,0,0] -> [1,1,1] in sim Y-up coords, mapped to
    [0,0,0] -> [1,1,1] in Blender Z-up (Y_sim is up, becomes Z_bl after the
    Y/Z swap in load_frame). Tank walls hug those bounds exactly so water can
    visibly press against the glass.
    """
    glass = make_glass_material()
    wood  = make_principled("Counter", (0.18, 0.11, 0.06), roughness=0.45)
    walle = make_principled("BackWall", (0.55, 0.45, 0.40), roughness=0.85)

    # Counter — wood slab the tank sits on. Top at z = -0.005 so its surface
    # is just below the tank's bottom face (no z-fighting).
    counter_size = 3.0
    _box("Counter",
         lo=(domain_size / 2 - counter_size / 2,
             domain_size / 2 - counter_size / 2,
             -0.05),
         hi=(domain_size / 2 + counter_size / 2,
             domain_size / 2 + counter_size / 2,
             -0.005),
         material=wood)

    # Back wall — vertical plane behind the tank. Camera looks toward -Y, so
    # this sits at +Y far side. (Camera setup will place camera at +Y, looking
    # back toward -Y, so back wall is "behind" from camera's POV).
    bpy.ops.mesh.primitive_plane_add(
        size=6.0,
        location=(domain_size / 2, domain_size + 2.5, 1.2),
        rotation=(math.radians(90), 0, 0))
    bw = bpy.context.active_object
    bw.name = "BackWall"
    bw.data.materials.append(walle)

    # Glass tank — 5 walls, no top. Wall thickness = 2cm. Tank interior
    # exactly matches sim domain.
    t = 0.02
    # Floor: just below sim floor
    _box("Tank.Floor",
         lo=(0.0, 0.0, -t),
         hi=(domain_size, domain_size, 0.0),
         material=glass)
    # +X wall (right)
    _box("Tank.PlusX",
         lo=(domain_size, 0.0, 0.0),
         hi=(domain_size + t, domain_size, domain_size),
         material=glass)
    # -X wall (left)
    _box("Tank.MinusX",
         lo=(-t, 0.0, 0.0),
         hi=(0.0, domain_size, domain_size),
         material=glass)
    # +Y wall (back of tank from camera POV — between camera and water at far side)
    _box("Tank.PlusY",
         lo=(0.0, domain_size, 0.0),
         hi=(domain_size, domain_size + t, domain_size),
         material=glass)
    # -Y wall (front — closest to camera)
    _box("Tank.MinusY",
         lo=(0.0, -t, 0.0),
         hi=(domain_size, 0.0, domain_size),
         material=glass)


def _aim_at(obj, target):
    """Add a Track-To constraint so the object's -Z axis points at target."""
    bpy.ops.object.empty_add(location=target)
    aim = bpy.context.active_object
    aim.name = obj.name + ".target"
    c = obj.constraints.new(type="TRACK_TO")
    c.target = aim
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"


def setup_world_hdri(hdri_path: str, strength: float = 1.0,
                      rotation_deg: float = 0.0):
    """World lit by an HDRI environment map.

    Replaces the gradient background. HDRIs give physically-grounded
    directional sun + soft sky ambient in one image — much closer to real
    product photography than a synthetic light rig, and the directional
    component drives clean caustics through the water.
    """
    bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    map_ = nt.nodes.new("ShaderNodeMapping")
    coord = nt.nodes.new("ShaderNodeTexCoord")

    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    map_.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)
    bg.inputs["Strength"].default_value = strength

    nt.links.new(coord.outputs["Generated"], map_.inputs["Vector"])
    nt.links.new(map_.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_lighting(domain_size=1.0, hdri_path: str = None,
                    hdri_strength: float = 1.0, hdri_rotation: float = 0.0):
    """Lighting + world background.

    With --hdri: HDRI carries the lighting; we add only a subtle warm rim
    light to keep the tank's silhouette readable against the environment.

    Without --hdri: full three-point area rig + gradient sky (legacy path).
    """
    target = (domain_size / 2, domain_size / 2, domain_size * 0.3)

    if hdri_path:
        setup_world_hdri(hdri_path, hdri_strength, hdri_rotation)
        # Single soft rim — most light is the HDRI; this just kicks the edge.
        bpy.ops.object.light_add(
            type="AREA",
            location=(domain_size / 2, -domain_size * 1.5, domain_size * 0.8))
        rim = bpy.context.active_object
        rim.name = "RimLight"
        rim.data.energy = 30
        rim.data.size = 1.0
        rim.data.color = (1.0, 0.92, 0.80)
        _aim_at(rim, target)
        return

    # ---- legacy path: three-point rig + gradient sky ----
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
    """3/4 angle on the tank, pulled back so the tank reads as an object."""
    target = (domain_size / 2, domain_size / 2, domain_size * 0.35)
    cam_pos = (domain_size * 1.7, -domain_size * 2.2, domain_size * 1.1)

    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.data.lens = 50
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


def load_water_mesh(ply_path: str, water_mat) -> bpy.types.Object:
    """Import a pre-reconstructed water surface PLY mesh.

    The PLY is built externally by tools/particles_to_mesh.py with
    coordinates ALREADY in Blender Z-up convention — no transform
    needed here. This gives much smoother surfaces than metaballs
    (which are inherently blobby).
    """
    try:
        bpy.ops.wm.ply_import(filepath=ply_path)
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    obj.name = "Fluid"
    obj.data.materials.append(water_mat)
    return obj


def make_metaball_cluster(pts, radius, water_mat):
    """Metaballs auto-merge into a fluid surface. Pts are already Z-up Blender.

    Tuned for smooth fluid surface (not visible particle bumps):
      - Large element radius (7x particle radius) so neighboring metaballs
        overlap heavily.
      - Low threshold (0.10) so the iso-surface forms close to where
        elements end, fusing them.
      - Fine resolution for the marching-cubes mesh extractor (sized to
        the actual particle radius, not the inflated metaball radius).
    """
    mball = bpy.data.metaballs.new("FluidMeta")
    mball.resolution        = radius * 0.4
    mball.render_resolution = radius * 0.25
    mball.threshold         = 0.10
    obj = bpy.data.objects.new("Fluid", mball)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(water_mat)
    for p in pts:
        e = mball.elements.new(type="BALL")
        e.co = p
        e.radius = radius * 7.0    # heavy overlap for full surface merging
    return obj


# ---------- top-level ----------

def render_one(out_path: str, frame_path: str = None,
                mesh_path: str = None,
                spp=64, res=(1280, 720), radius=0.012, domain_size=1.0,
                no_water=False, hdri_path: str = None,
                hdri_strength: float = 1.0, hdri_rotation: float = 0.0,
                caustics: bool = True):
    reset_scene()
    setup_render(spp, res[0], res[1], caustics=caustics)
    setup_room(domain_size)
    setup_lighting(domain_size, hdri_path=hdri_path,
                   hdri_strength=hdri_strength, hdri_rotation=hdri_rotation)
    setup_camera(domain_size)

    if no_water:
        print("  --no-water: rendering empty room")
    elif mesh_path is not None:
        print(f"  loading water mesh from {mesh_path}")
        load_water_mesh(mesh_path, make_water_material())
    elif frame_path is not None:
        pts = load_frame(frame_path)
        print(f"  loaded {len(pts)} particles from {frame_path}")
        make_metaball_cluster(pts, radius, make_water_material())
    else:
        sys.exit("must supply frame_path, mesh_path, or no_water=True")

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
    p.add_argument("--frame",     help="single binary frame")
    p.add_argument("--seq",       help="directory of frame_NNNN.bin (batch)")
    p.add_argument("--mesh",      help="single PLY mesh from particles_to_mesh.py")
    p.add_argument("--mesh-seq",  help="directory of frame_NNNN.ply (batch)")
    p.add_argument("--out",       help="single output PNG")
    p.add_argument("--out-dir",   help="batch output directory")
    p.add_argument("--spp", type=int, default=256,
                   help="Cycles samples per pixel (caustics need >=128)")
    p.add_argument("--res", nargs=2, type=int, default=[1280, 720])
    p.add_argument("--radius", type=float, default=0.012)
    p.add_argument("--domain", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-water", action="store_true",
                   help="render the room only, no fluid")
    p.add_argument("--hdri", help="path to .hdr/.exr environment map")
    p.add_argument("--hdri-strength", type=float, default=1.0,
                   help="HDRI emission strength multiplier")
    p.add_argument("--hdri-rotation", type=float, default=0.0,
                   help="HDRI yaw rotation in degrees (rotates sun direction)")
    p.add_argument("--no-caustics", action="store_true",
                   help="disable refractive caustics (faster, no light bending)")
    args = p.parse_args(argv)

    common = dict(
        spp=args.spp,
        res=tuple(args.res),
        radius=args.radius,
        domain_size=args.domain,
        hdri_path=args.hdri,
        hdri_strength=args.hdri_strength,
        hdri_rotation=args.hdri_rotation,
        caustics=not args.no_caustics,
    )

    if args.no_water:
        if not args.out:
            sys.exit("--out required with --no-water")
        render_one(args.out, frame_path=None, no_water=True, **common)
        return

    if args.mesh:
        if not args.out:
            sys.exit("--out required with --mesh")
        render_one(args.out, mesh_path=args.mesh, **common)
    elif args.mesh_seq:
        if not args.out_dir:
            sys.exit("--out-dir required with --mesh-seq")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        files = sorted(Path(args.mesh_seq).glob("frame_*.ply"))[::args.stride]
        for i, f in enumerate(files):
            out = Path(args.out_dir) / f"render_{f.stem.split('_')[1]}.png"
            print(f"[{i+1}/{len(files)}] {f.name} -> {out.name}")
            render_one(str(out), mesh_path=str(f), **common)
    elif args.frame:
        if not args.out:
            sys.exit("--out required with --frame")
        render_one(args.out, frame_path=args.frame, **common)
    elif args.seq:
        if not args.out_dir:
            sys.exit("--out-dir required with --seq")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        files = sorted(Path(args.seq).glob("frame_*.bin"))[::args.stride]
        for i, f in enumerate(files):
            out = Path(args.out_dir) / f"render_{f.stem.split('_')[1]}.png"
            print(f"[{i+1}/{len(files)}] {f.name} -> {out.name}")
            render_one(str(out), frame_path=str(f), **common)
    else:
        sys.exit("must pass --frame, --seq, --mesh, --mesh-seq, or --no-water")


if __name__ == "__main__":
    main()

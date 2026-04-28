# water_sim

GPU-native water simulation and offline cinematic renderer.
Modern rebuild of a 2023 bachelor project.

## Status

- ✅ **Phase 1 (Foundation)** — modern toolchain wired (CUDA 13.2 + Vulkan 1.4 + Slang + LBVH + JSON scene loader + test harness)
- ✅ **Phase 2 (DFSPH solver)** + **Phase 2.5 (boundaries + viscosity + convergence)** — DFSPH with Akinci 2012 boundary particles, XSPH viscosity, CUB-based density-solver convergence, and explicit damping. Settled-puddle regression test green.
- ✅ **Cycles render pipeline** (Blender) — photoreal path-traced glass-tank scene via `tools/blender_render.py`. Reads sim binary frames → metaball fluid surface → renders with proper Z-up convention, glass tank visible at sim domain bounds, wood counter, soft three-point lighting.
- ⏳ **Phase 3** (in-engine screen-space viewport) — not started
- ⏳ **Phase 4** (anisotropic surface reconstruction in CUDA) — not started
- ⏳ **Phase 5** (Vulkan-RT path tracer) — not started; rendering currently bridged via Blender Cycles
- ⏳ **Phase 6** (OIDN denoise + ffmpeg cinematic mux) — partially via ffmpeg; no in-renderer denoise

See [`docs/superpowers/specs/2026-04-23-water-simulation-rebuild-design.md`](docs/superpowers/specs/2026-04-23-water-simulation-rebuild-design.md) for the full design and [`docs/superpowers/plans/`](docs/superpowers/plans/) for per-phase implementation plans.

## What works end-to-end (today)

```bash
# Simulate
./build/linux-debug-local/bin/sim_cli \
    --scene scenes/dam_break.json --frames 0:90 \
    --record --no-vulkan --out out/dam_break

# Render with Cycles (glass-tank scene)
blender --background --python tools/blender_render.py -- \
    --seq out/dam_break --out-dir out/dam_break_render \
    --stride 2 --spp 64 --res 1280 720

# Mux to mp4
ffmpeg -y -framerate 15 -pattern_type glob \
    -i 'out/dam_break_render/render_*.png' \
    -c:v libx264 -pix_fmt yuv420p -crf 18 out/dam_break.mp4
```

Produces a photoreal animation of a glass column collapsing inside a glass tank on a wood counter, with proper splash dynamics, refraction, and settling.

## Tests

26 unit tests + 2 regression tests, all green:
- `water_tests` — kernels, particle store, cell grid, LBVH, scene loader, boundary sampler, DFSPH density/α/gravity
- `water_regression` — diagnostic (single-particle gravity + boundary mass) + settled puddle (4913-particle block falls into 1m³ box, settles into stable puddle in <1.5s sim time, no NaN, no escapees)

## Tech stack

CUDA 13.2 · Vulkan 1.4 + ray tracing extensions · Slang shaders → SPIR-V · CMake 3.27 · C++20 · CCCL (Thrust + CUB + libcu++) · doctest · nlohmann/json.

No OpenGL anywhere.

## Hardware target

Primary: **NVIDIA RTX 5070 Mobile** (Blackwell, sm_120, 8 GB VRAM). Should also work on any RTX with VK_KHR_ray_tracing support; override `CMAKE_CUDA_ARCHITECTURES` for non-Blackwell GPUs.

## Building

### Prerequisites
- CUDA Toolkit 13.x
- Vulkan SDK 1.4 (LunarG; on Fedora `sudo dnf install vulkan-devel vulkan-tools vulkan-validation-layers-devel`)
- `slangc` 2026.x on PATH (download from [shader-slang releases](https://github.com/shader-slang/slang/releases))
- gcc ≤ 15 *or* clang as the CUDA host compiler (CUDA 13.2 doesn't support gcc 16; on Fedora 44 with gcc 16, use a conda gcc-14 toolchain — see `CMakeUserPresets.json` template).
- CMake 3.27+ and Ninja

### Build & test
```bash
# If you needed to override the CUDA host compiler, copy the template:
#   (see "CMakeUserPresets.json" in repo root for an example — gitignored)

cmake --preset linux-debug                 # or linux-debug-local if using user preset
cmake --build build/linux-debug -j
ctest --preset linux-debug
./build/linux-debug/bin/sim_cli --scene scenes/single_drop.json
```

Expected smoke-test output:
```
=== water_sim sim_cli ===
scene:        single_drop
output:       320x240 @ 24.0 fps, frames 0..4
solver:       dfsph
particle r:   0.0100 m
rest density: 1000.0 kg/m^3
vulkan:       NVIDIA GeForce RTX 5070 Laptop GPU (api 1.4.x, RT yes)
particles:    180 (initial block)
lbvh leaves:  180
first dt:     0.004000 s (CFL with v=1 m/s)
=== sim_cli OK ===
```

## Repository layout

```
water_simulation/
├── core/      # CUDA particle store, LBVH, boundary, timestepper
├── scene/     # JSON scene loader
├── renderer/  # Vulkan 1.4 device + Slang shader build pipeline
├── apps/      # sim_cli (Phase 1) + viewport (Phase 3) entry points
├── tests/     # doctest unit tests
├── scenes/    # example scene JSON files
├── docs/      # design spec + per-phase implementation plans
└── legacy/    # original 2023 SPH+OpenGL code preserved as-is
```

## License
TBD — currently personal portfolio.

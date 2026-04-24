# water_sim

GPU-native water simulation and offline cinematic renderer.
Modern rebuild of a 2023 bachelor project.

## Status

- ✅ **Phase 1 (Foundation)** complete. Modern toolchain wired — CUDA 13.2 + Vulkan 1.4 + Slang + LBVH spatial accel + JSON scene loader + test harness.
- ⚠️ **Phase 2 (DFSPH solver)** *partially complete and not yet stable*. CellGrid, SPH kernels, density + α factor + density-invariance iteration scaffolding all build and unit-test green, but the solver is **not stable for multi-particle cinematic output** because boundary handling is a soft AABB clamp + bounce, not proper boundary forces. Water spreads infinitely thin instead of stacking. See [`docs/superpowers/plans/2026-04-24-water-simulation-v2-phase2.5-boundaries-and-viscosity.md`](docs/superpowers/plans/2026-04-24-water-simulation-v2-phase2.5-boundaries-and-viscosity.md) for the focused fix-up plan.
- ⏳ **Phase 3** (screen-space dev viewport) — not started
- ⏳ **Phase 4** (anisotropic surface reconstruction) — not started
- ⏳ **Phase 5** (Vulkan-RT path tracer) — not started
- ⏳ **Phase 6** (OIDN denoise + ffmpeg cinematic mux) — not started

See [`docs/superpowers/specs/2026-04-23-water-simulation-rebuild-design.md`](docs/superpowers/specs/2026-04-23-water-simulation-rebuild-design.md) for the full design and [`docs/superpowers/plans/`](docs/superpowers/plans/) for per-phase implementation plans.

### Honest limitations of the current Phase 2 partial

What works:
- All 23 unit tests pass (CellGrid, SPH kernels, density/α correctness in controlled cases, single-particle gravity-then-floor)
- `sim_cli` runs scenes end-to-end and dumps per-frame particle positions
- `tools/viz.py` renders mp4 from the dumps
- A dam-break visually shows column collapse and surge dynamics

What doesn't work (yet):
- **Settled fluid does not stack** — water spreads to a single-layer film on the floor instead of forming a puddle of correct depth
- **Pressure is not zero at rest** — even at rest spacing, density samples to ρ₀ but boundary undersampling produces spurious pressure gradients
- **No viscosity** — fluid is inviscid, contributing to chaotic motion
- **Density solver fixed at 20 iters** — no convergence-based early exit
- **`max_velocity` is a constant placeholder** — adaptive CFL substepping is not really adaptive

These are addressed by Phase 2.5 (Akinci 2012 boundary particles + viscosity + density convergence).

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

# Water Simulation v2 — Design Spec

**Date:** 2026-04-23
**Author:** Frank Yin (with Claude Code)
**Status:** Draft → pending user review
**Supersedes:** the original 2023 bachelor-project SPH+OpenGL implementation

---

## 1. Overview

Rebuild the water simulation as a modern GPU-native offline cinematic renderer. Target deliverable for v1 is a 3–5 second 1080p video of water pouring into a glass, rendered offline with ray-traced caustics and surface tension effects.

The original 2023 codebase (WCSPH solver + OpenGL particle viewer + unfinished Marching Cubes) is preserved under `legacy/` for reference and historical interest. The new architecture replaces it entirely.

### 1.1 Why rebuild

- **WCSPH visibly compresses water under impact** — disqualifying for cinematic quality
- **OpenGL is in maintenance mode** — modern GPU features (RT cores, mesh shaders, bindless) are awkward or absent
- **Surface reconstruction was never finished** — the actual rendering bottleneck for SPH water
- **CUDA architecture is 2014-era** — no graphs, no async memory, no LBVH, no Slang/CCCL/libcu++

### 1.2 Success criteria (v1)

- `out/glass_pour_v1.mp4` exists: 3–5 sec, 1080p, 24 fps, h.264
- Visible caustics on the wood surface beneath the glass
- Visible surface tension: droplet pinch-off in pour stream, rim beading, meniscus
- Clean convergence at 2048 SPP with OIDN denoising
- Total render: ≤ 12 hours overnight on the RTX 5070 Mobile (8 GB VRAM)
- Architecture supports adding FLIP solver and direct-implicit surface tracing in v2 without touching the renderer or scene system

### 1.3 Non-goals (v1)

- Real-time rendering performance
- Colored / multi-fluid simulation
- Subsurface scattering (water is clear)
- Wet-glass exterior droplets / condensation
- Dispersion / rainbow caustics
- Interactive camera control during render
- Windows build parity (Linux-first; Windows comes later if at all)
- Cross-vendor GPU testing (NVIDIA-only is acceptable in v1)

---

## 2. Hardware target & constraints

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop (Blackwell, sm_120) |
| VRAM | 8151 MiB (~8 GB usable, ~6 GB working set after OS/driver overhead) |
| Driver | 595.58.03 |
| CUDA | 13.2 |
| Vulkan API | 1.4.329 |
| OS | Linux (Fedora 44 confirmed; Ubuntu/Arch should work identically) |
| RT cores | 4th-generation (Blackwell) |
| Tensor cores | 5th-generation (FP4 capable) |

The 8 GB VRAM is the binding constraint. Memory budget (§11) is sized accordingly.

---

## 3. Tech stack

| Layer | Technology | Rationale |
|---|---|---|
| Simulation | CUDA 13.2 C++20 | Existing competence, mature ecosystem, RT cores accessible via OptiX/Vulkan interop |
| GPU primitives | CCCL (Thrust + CUB + libcu++) | Modern unified umbrella replacing ad-hoc Thrust calls |
| Memory | `cudaMallocAsync` + memory pools | Stream-ordered, fewer global syncs |
| Kernel reuse | CUDA Graphs | Record entire timestep once, replay per frame |
| Spatial accel | LBVH (Karras 2012) via Morton codes | Replaces cell-grid hash; reusable for ray tracing |
| Rendering | Vulkan 1.4 + VK_KHR_ray_tracing_pipeline | Modern, hardware-accelerated, vendor-neutral |
| Shaders | Slang (Khronos) | Generics, modules, multi-target, auto-diff future-proof |
| Display | Vulkan + CUDA-Vulkan interop (external memory) | One API for display + RT, no OpenGL |
| Denoising | Intel OIDN 2.x | Cross-platform, high quality, runs on CUDA backend |
| Image I/O | OpenEXR + libpng + ffmpeg subprocess | EXR for working frames, PNG for previews, mp4 for delivery |
| Math | GLM (CPU-side), `cuda::std::array`/custom (GPU-side) | Avoid GLM on device; use small custom vec types matching CUDA conventions |
| Scene format | JSON via nlohmann/json | Declarative, human-readable, no recompile to change shot |
| Build | CMake 3.27+ with first-class CUDA language | Replaces nvcc-via-`add_custom_command` hack |
| Deps | FetchContent (header-only) + system Vulkan SDK | Keeps repo self-contained where possible |
| Profiling | Nsight Systems (timeline) + Nsight Compute (kernel) + Tracy (CPU side) | Modern GPU profiling, not printf |
| Tests | doctest (header-only) + custom regression harness | Lightweight, no build-system burden |

---

## 4. Top-level architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  SCENE (declarative: scenes/glass_pour.json + assets/)             │
│  camera, lights, HDRI, materials, solver config, output settings   │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  CORE (solver-agnostic, CUDA)                                      │
│  ParticleStore │ SpatialAccel (LBVH) │ Boundary (SDF) │ TimeStep   │
└────────────────────────────────────────────────────────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
│  SOLVER      │         │  SURFACE     │         │  IO              │
│  (CUDA)      │         │  RECON       │         │  checkpointing,  │
│              │         │  (CUDA)      │         │  EXR/PNG writer, │
│  DFSPH v1    │─────────│  Aniso+MC    │         │  ffmpeg mux      │
│  (FLIP v2)   │         │  (Implicit   │         │                  │
│              │         │   v2)        │         │                  │
└──────────────┘         └──────────────┘         └──────────────────┘
                                │
                                ▼  mesh + attribs, per frame
                ┌───────────────────────────────────┐
                │  RENDERER (Slang → SPIR-V,        │
                │  Vulkan 1.4 + VK_KHR_ray_tracing) │
                │                                   │
                │  Path tracer w/ ReSTIR for        │
                │  caustics, OIDN denoise, camera   │
                │  (DoF + motion blur), BSDFs       │
                └───────────────────────────────────┘
                                │
                                ▼
                          frames/*.exr → ffmpeg → .mp4

Parallel dev viewport (separate binary):
┌──────────────────────────────────┐
│ Screen-space fluids (Slang       │
│ compute in Vulkan), CUDA-Vulkan  │
│ interop on particle buffer       │
└──────────────────────────────────┘
```

### 4.1 Module boundaries

The boundary between **simulation** and **rendering** is a per-frame **handoff buffer**: a Vulkan-imported CUDA buffer containing the reconstructed surface mesh (vertices, normals, indices). The simulation writes; the renderer consumes to build the BLAS. No CPU roundtrip.

The boundary between **solver-specific** and **solver-agnostic** code is the `Solver` interface (§6). Even though only DFSPH exists in v1, the module structure assumes another solver could replace it.

The boundary between **dev viewport** and **offline render** is total — they're separate binaries (`viewport` and `render`) sharing only the simulation core. The viewport is screen-space fluids for fast iteration; the renderer is path tracing for final frames.

---

## 5. Tech stack philosophy: what's ruthlessly modern, what's pragmatically conservative

**Ruthlessly modern (no compromises):**
- Vulkan 1.4 + ray tracing extensions (no OpenGL)
- Slang shaders (no GLSL)
- CCCL / libcu++ (no ad-hoc CUDA primitives)
- LBVH (no cell-grid hashing)
- DFSPH (no WCSPH)
- Anisotropic kernel reconstruction (no isotropic blob soup)
- CMake first-class CUDA (no nvcc shimming)
- JSON scene format (no compiled-in scene)

**Pragmatically conservative:**
- Triangle BLAS for v1 (procedural primitives in v2)
- Marching Cubes mesh extraction (direct implicit ray tracing in v2)
- Unidirectional path tracing + ReSTIR DI/GI (BDPT/manifold-NEE in v2 if caustics need it)
- OIDN post-process denoising (no in-flight neural denoising)
- Single-fluid (multi-fluid coupling is v3)
- TOML rejected in favor of JSON for ubiquity

---

## 6. Module specifications

### 6.1 `core/particle_store`

**Purpose:** Extensible structure-of-arrays particle attribute store. Particles live in CUDA device memory; layout is solver-driven (DFSPH declares `density`, `alpha`, etc.; FLIP would declare different attributes).

**Interface (sketch):**
```cpp
class ParticleStore {
public:
    enum class AttribType { F32, F32x3, F32x9, U32 };

    explicit ParticleStore(uint32_t capacity);

    // Solvers register the attributes they need at construction.
    template<typename T>
    AttribHandle<T> register_attribute(std::string_view name);

    // Resize (active particle count); capacity is fixed at construction.
    void resize(uint32_t count);
    uint32_t count() const noexcept;
    uint32_t capacity() const noexcept;

    // Typed, bounds-checked access.
    template<typename T>
    cuda::std::span<T> view(AttribHandle<T> h);
};
```

**Implementation notes:**
- Each attribute is an independently-allocated `cudaMallocAsync` buffer
- Built-in attributes (always present): `position` (float3), `velocity` (float3)
- Extension attributes registered at solver construction
- No per-particle struct; pure SoA for coalesced GPU access
- Span views enforce bounds in debug builds; raw pointers available for kernels

### 6.2 `core/spatial_accel`

**Purpose:** GPU-resident LBVH for radius-r neighbor queries. Built per-frame from particle positions.

**Algorithm (Karras 2012 + Aila/Laine 2009 traversal):**
1. Compute Morton codes for each particle position (CUB)
2. Sort particles by Morton code (CUB device-wide radix sort)
3. Build internal node hierarchy in parallel (Karras's algorithm — O(N) work)
4. Compute per-node AABBs bottom-up
5. Expose `for_each_neighbor(idx, radius, lambda)` device function for kernel use

**Why LBVH over cell-grid:**
- Adaptive to particle density (cell grid wastes memory in low-density regions)
- Same data structure used for ray tracing in v2 (procedural primitive intersection)
- Faster build than full SAH BVH; nearly as good traversal for SPH neighborhood queries

**Build cost:** ~0.2 ms for 800K particles on RTX 5070 (target). Per substep.

### 6.3 `core/boundary`

**Purpose:** Static colliders represented as signed distance fields (SDFs) sampled on a uniform grid.

**Why SDF (not boundary particles):**
- Smooth, no discretization artifacts at the rim
- Cheap query: trilinear interpolation
- Decouples boundary geometry resolution from particle resolution
- Glass surface gets clean meniscus

**Loading:** Mesh → SDF baked at scene load via CPU (libigl-style winding-number method or fast sweep). Stored as 3D texture for fast GPU sampling.

**Boundary forces:** Akinci 2012 boundary handling adapted for SDF — sample SDF gradient, push particles outward, dampen normal velocity component.

### 6.4 `core/timestep`

**Purpose:** Adaptive substepping with CFL-based dt selection, checkpoint scheduling.

**Logic:**
- Per-frame target: 1/24 s = 41.67 ms simulation time
- Substep dt: `min(dt_max, lambda * h / max_velocity)` with `lambda = 0.4`
- Substep count clamped to `[1, 16]`; alarm if hit ceiling
- Every Nth substep, snapshot particle state to disk for restart/regression

### 6.5 `solver/dfsph`

**Purpose:** Divergence-Free SPH solver (Bender & Koschier 2015) with surface tension (Akinci 2013) and air bubble extension (Ihmsen 2011, optional in v1).

**Per-substep pipeline:**
1. **Neighbor search:** rebuild LBVH (or refit if config allows)
2. **Density computation:** sum kernel contributions from neighbors
3. **Compute α factors:** `α_i = ρ_i / Σ_j ||∇W_ij m_j||²` (DFSPH's key precomputation)
4. **Apply external forces:** gravity, surface tension (Akinci 2013), air pressure
5. **Density-invariance solver:** iterative Jacobi-style correction loop
   - Predict density change
   - Compute pressure from `(ρ_predicted - ρ_0) * α / dt²`
   - Apply pressure to velocity
   - Repeat until `mean(ρ_predicted - ρ_0) < eta_density` (typically eta=1e-3)
   - Cap at 100 iterations; warn on hit
6. **Advect positions:** `x += v * dt`
7. **Divergence-free solver:** similar iterative loop on `∇·v`
   - Constraint: `Dρ/Dt = 0` (no inflow/outflow at material surfaces)
   - Cap at 100 iters

**Surface tension model (Akinci 2013):**
- Cohesion forces between fluid neighbors
- Curvature minimization via color field gradient
- Adhesion to boundary particles

**Stability:** DFSPH is robust to dt 10-100× larger than WCSPH. Expected substep count for 1/24s frame: 4-8.

### 6.6 `surface/aniso_mc`

**Purpose:** Reconstruct fluid surface mesh from particles for ray tracing.

**Pipeline (Yu & Turk 2013 + standard Marching Cubes):**
1. **Per-particle anisotropy:** for each particle, gather neighbors weighted by smoothing kernel. Compute weighted covariance matrix. SVD → principal axes. Build anisotropic transform `G_i` (3×3 matrix per particle, scales kernel along principal flow direction).
2. **Smooth particle positions:** weighted average of neighbor positions (Yu-Turk's `x̃_i`).
3. **Density grid:** for each grid cell, sum anisotropic kernel contributions from nearby particles (queried via LBVH).
4. **Marching Cubes:** classic 1987 algorithm on the density grid. Output: vertex+index buffer. Use the pre-baked LUT (already in `legacy/include/MarchingCubesLUT.h`).
5. **Mesh upload:** write to Vulkan-imported CUDA buffer; renderer rebuilds BLAS from this.

**Grid resolution:** 384³ for v1 (memory-constrained on 8GB VRAM; see §11). Allocates ~225 MB for the float density grid.

**Output mesh size estimate:** 1-3M triangles per frame for the glass-pour scene. Acceptable for BLAS build (~50-100 ms on RT cores).

### 6.7 `render/vk`

**Purpose:** Vulkan 1.4 device/swapchain/pipeline scaffolding. Owns the RT pipeline, descriptor sets, command buffer recording.

**Components:**
- `vk::Instance` — Vulkan 1.4 instance with debug utils, ray tracing extensions
- `vk::Device` — device handle, queue families (graphics+compute+transfer), VMA allocator
- `vk::Swapchain` — only for dev viewport; offline render writes directly to imported buffers
- `vk::AccelerationStructure` — TLAS + BLAS for the fluid mesh and static scene geometry
- `vk::RayTracingPipeline` — shader binding table, hit groups
- `vk::CudaInterop` — `VK_KHR_external_memory` import of CUDA buffers (mesh, denoiser input)

**Required extensions:**
- `VK_KHR_ray_tracing_pipeline`
- `VK_KHR_acceleration_structure`
- `VK_KHR_buffer_device_address`
- `VK_KHR_external_memory_fd`
- `VK_KHR_external_semaphore_fd`
- `VK_EXT_descriptor_indexing` (bindless)
- `VK_KHR_synchronization2`

### 6.8 `render/shaders` (Slang)

**Purpose:** All render-side compute and ray tracing shaders, written in Slang, compiled to SPIR-V.

**Shader inventory for v1:**
- `raygen.slang` — primary ray generation, camera (with DoF + motion blur sampling)
- `miss.slang` — environment HDRI sample
- `hit_glass.slang` — closest-hit for the glass material (rough dielectric BSDF)
- `hit_water.slang` — closest-hit for the fluid surface mesh (water dielectric, IOR 1.33)
- `hit_diffuse.slang` — closest-hit for wood surface (Disney diffuse + microfacet specular)
- `bsdfs.slang` — module: BSDF library (rough dielectric, Disney diffuse, GGX microfacet)
- `sampling.slang` — module: hemisphere sampling, Mitchell-Netravali filtering, Halton sequences
- `restir_di.slang` — ReSTIR direct illumination temporal+spatial reuse
- `accumulate.slang` — frame accumulation (per-pixel running mean)
- `tonemap.slang` — ACES tone mapping for preview output

**Slang module layout:** shared math/sampling/BSDF code factored into modules; entry shaders import them. Builds via `slangc` invoked from CMake.

### 6.9 `render/denoise`

**Purpose:** OIDN integration for post-process denoising of accumulated frames.

**Approach:**
- Render at 2048 SPP into a 32-bit float HDR accumulation buffer
- Generate aux buffers: albedo, normal (first-bounce surface attributes)
- After convergence per frame: copy to OIDN, denoise with `oidn::FilterRT`, write to EXR
- ffmpeg consumes EXR sequence → mp4

OIDN's CUDA backend keeps data on-GPU; no CPU roundtrip during denoising.

### 6.10 `viewport/ssfluid`

**Purpose:** Real-time dev viewport. Renders particles as screen-space fluid (Müller 2007 / van der Laan 2009) for fast iteration without going through the offline path.

**Pipeline (all Slang compute):**
1. Render particles as depth points (point sprites or compute-driven)
2. Bilateral depth filter (smooths surface while preserving edges)
3. Reconstruct world-space normals from filtered depth
4. Composite with simple Phong/PBR shading + environment cubemap reflection
5. Display via Vulkan swapchain

**Features:**
- ImGui overlay for stats (substep count, particle count, ms/frame)
- Camera controls (similar to old code: WASD + mouse look)
- Hot-reload of solver parameters from JSON file watched on disk
- Pause / single-step / restart

### 6.11 `scene/loader`

**Purpose:** Parse JSON scene → in-memory representation; load referenced assets (HDRI, meshes).

**Responsibilities:**
- Validate JSON against an internal schema
- Resolve asset paths relative to scene file
- Bake collider meshes to SDFs (cached on disk, hash-keyed by mesh + grid params)
- Materialize CUDA particle initial positions (initial fluid block, emitters)

### 6.12 `io/frame_writer`

**Purpose:** Write rendered frames to disk; mux to video.

- EXR via OpenEXR (32-bit float, full HDR)
- PNG via libpng (sRGB-tonemapped, for quick previews)
- ffmpeg invoked as subprocess at end of run for mp4 mux
- Per-frame metadata sidecar JSON (camera, lighting, sim state ref)

### 6.13 `app/sim_cli` and `app/viewport`

Two binaries:
- **`app/sim_cli`**: headless. Loads scene, runs sim + render to disk, exits.
  ```
  ./sim_cli --scene scenes/glass_pour.json --frames 0:120 --spp 2048 --out out/glass_pour
  ```
- **`app/viewport`**: interactive. Loads scene, runs sim, displays via screen-space fluids.
  ```
  ./viewport --scene scenes/glass_pour.json
  ```

---

## 7. Scene format (JSON)

Example scene file:

```json
{
  "schema_version": "1.0",
  "name": "glass_pour",
  "output": {
    "resolution": [1920, 1080],
    "fps": 24,
    "frame_range": [0, 120],
    "format": "exr",
    "video": "mp4"
  },
  "camera": {
    "type": "thin_lens",
    "position": [0.0, 0.4, 1.2],
    "look_at": [0.0, 0.15, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_y_deg": 30,
    "aperture_radius": 0.005,
    "focus_distance": 1.2,
    "shutter_open_close": [0.0, 1.0],
    "motion": {
      "type": "dolly",
      "duration_sec": 5.0,
      "end_position": [0.0, 0.3, 0.6],
      "easing": "ease_in_out"
    }
  },
  "lighting": {
    "environment": {
      "hdri": "assets/hdri/studio_softbox_4k.exr",
      "intensity": 1.0,
      "rotation_deg": 45
    },
    "lights": [
      {
        "type": "rect_area",
        "position": [-1.0, 1.5, 0.5],
        "size": [0.8, 0.8],
        "rotation_deg": [-45, 30, 0],
        "color": [1.0, 0.95, 0.9],
        "intensity": 100.0
      }
    ]
  },
  "materials": {
    "glass": {
      "type": "rough_dielectric",
      "ior": 1.50,
      "roughness": 0.02,
      "transmission_color": [0.99, 0.99, 0.99]
    },
    "water": {
      "type": "rough_dielectric",
      "ior": 1.33,
      "roughness": 0.0,
      "transmission_color": [0.95, 0.97, 1.0]
    },
    "wood": {
      "type": "disney_principled",
      "base_color_tex": "assets/textures/dark_walnut_albedo.exr",
      "roughness": 0.4,
      "specular": 0.5
    }
  },
  "scene": {
    "objects": [
      {
        "name": "wood_surface",
        "type": "plane",
        "size": [2.0, 2.0],
        "position": [0.0, 0.0, 0.0],
        "material": "wood"
      },
      {
        "name": "glass",
        "type": "mesh",
        "asset": "assets/meshes/water_glass.obj",
        "position": [0.0, 0.0, 0.0],
        "material": "glass",
        "is_collider": true
      }
    ],
    "fluid": {
      "solver": "dfsph",
      "particle_radius": 0.0015,
      "rest_density": 1000.0,
      "viscosity": 0.001,
      "surface_tension": 0.0728,
      "emitters": [
        {
          "type": "stream",
          "position": [0.0, 0.5, 0.0],
          "direction": [0.0, -1.0, 0.0],
          "radius": 0.008,
          "velocity": 1.5,
          "active_time": [0.0, 4.0]
        }
      ],
      "solver_params": {
        "max_density_iters": 100,
        "max_divergence_iters": 100,
        "eta_density": 1e-3,
        "eta_divergence": 1e-3,
        "cfl_lambda": 0.4
      }
    }
  },
  "render": {
    "spp": 2048,
    "max_bounces": 12,
    "russian_roulette_start": 4,
    "denoiser": "oidn",
    "tone_map": "aces"
  }
}
```

---

## 8. Build system

CMake 3.27+ with first-class CUDA language. Targets Linux first (Windows v2).

### 8.1 Top-level structure

```cmake
cmake_minimum_required(VERSION 3.27)
project(water_sim VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)  # Blackwell

# Modules
add_subdirectory(third_party)  # FetchContent: glm, json, doctest, slang, oidn
add_subdirectory(core)
add_subdirectory(solvers)
add_subdirectory(surface)
add_subdirectory(renderer)
add_subdirectory(viewport)
add_subdirectory(scene)
add_subdirectory(io)
add_subdirectory(apps)
add_subdirectory(tests)
```

### 8.2 Dependency strategy

**FetchContent** (pinned versions):
- glm 1.0.1 (CPU-side math)
- nlohmann/json 3.11.x (scene loader)
- doctest 2.4.x (tests)
- stb (image read for textures)
- tinyobjloader (mesh load)

**System / pre-installed:**
- Vulkan SDK 1.4.x (LunarG)
- Slang compiler (`slangc` on PATH)
- CUDA 13.2 toolkit
- Intel OIDN 2.x (system package or FetchContent)
- ffmpeg (subprocess)

**Why mix:** small header-only deps go in FetchContent for hermeticity. Heavy SDKs (Vulkan, CUDA, OIDN) are system-installed because users running this almost certainly have them already, and FetchContent for a 2 GB SDK is rude.

### 8.3 Slang shader compilation

Custom CMake function:
```cmake
add_slang_shaders(
  TARGET render_shaders
  SHADERS
    renderer/shaders/raygen.slang
    renderer/shaders/miss.slang
    ...
  OUT_DIR ${CMAKE_BINARY_DIR}/shaders
  PROFILE sm_6_5  # SPIR-V target
  STAGES rgen miss chit ...
)
```

Calls `slangc -profile sm_6_5 -target spirv -o ...` per shader. Output `.spv` files copied to runtime directory.

### 8.4 CMake presets

Provide `CMakePresets.json`:
- `linux-debug` — `-G "Ninja" -DCMAKE_BUILD_TYPE=Debug`
- `linux-release` — `-G "Ninja" -DCMAKE_BUILD_TYPE=Release`
- `linux-relwithdebinfo` — for profiling

---

## 9. Directory layout

```
water_simulation/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── docs/
│   └── superpowers/specs/2026-04-23-water-simulation-rebuild-design.md  ← this file
├── core/
│   ├── CMakeLists.txt
│   ├── include/water/core/
│   │   ├── particle_store.h
│   │   ├── spatial_accel.h
│   │   ├── boundary.h
│   │   └── timestep.h
│   └── src/
│       ├── particle_store.cu
│       ├── spatial_accel.cu      # LBVH
│       ├── boundary.cu           # SDF sampling
│       └── timestep.cpp
├── solvers/
│   ├── CMakeLists.txt
│   └── dfsph/
│       ├── include/water/solvers/dfsph.h
│       └── src/
│           ├── dfsph.cu
│           ├── dfsph_density.cu
│           ├── dfsph_pressure.cu
│           ├── dfsph_surface_tension.cu
│           └── kernels.cuh        # SPH kernels (cubic spline, surface tension)
├── surface/
│   ├── CMakeLists.txt
│   └── aniso_mc/
│       ├── include/water/surface/aniso_mc.h
│       └── src/
│           ├── anisotropy.cu      # Yu-Turk SVD per-particle
│           ├── density_field.cu   # grid sampling
│           ├── marching_cubes.cu
│           └── mc_lut.h           # ported from legacy
├── renderer/
│   ├── CMakeLists.txt
│   ├── include/water/renderer/
│   │   ├── vk_device.h
│   │   ├── vk_acceleration.h
│   │   ├── vk_pipeline.h
│   │   ├── vk_cuda_interop.h
│   │   └── denoise.h
│   ├── src/
│   │   ├── vk_device.cpp
│   │   ├── vk_acceleration.cpp
│   │   ├── vk_pipeline.cpp
│   │   ├── vk_cuda_interop.cpp
│   │   └── denoise.cpp
│   └── shaders/
│       ├── raygen.slang
│       ├── miss.slang
│       ├── hit_glass.slang
│       ├── hit_water.slang
│       ├── hit_diffuse.slang
│       ├── modules/
│       │   ├── bsdfs.slang
│       │   ├── sampling.slang
│       │   └── camera.slang
│       └── post/
│           ├── accumulate.slang
│           └── tonemap.slang
├── viewport/
│   ├── CMakeLists.txt
│   ├── include/water/viewport/ssfluid.h
│   ├── src/
│   │   ├── ssfluid.cpp
│   │   ├── camera_ctrl.cpp
│   │   └── imgui_overlay.cpp
│   └── shaders/
│       ├── particle_depth.slang
│       ├── bilateral_filter.slang
│       └── ssfluid_shade.slang
├── scene/
│   ├── CMakeLists.txt
│   ├── include/water/scene/scene.h
│   └── src/
│       ├── loader.cpp
│       ├── sdf_baker.cpp
│       └── asset_cache.cpp
├── io/
│   ├── CMakeLists.txt
│   ├── include/water/io/frame_writer.h
│   └── src/
│       ├── exr_writer.cpp
│       ├── png_writer.cpp
│       └── ffmpeg_mux.cpp
├── apps/
│   ├── CMakeLists.txt
│   ├── sim_cli/
│   │   └── main.cpp
│   └── viewport/
│       └── main.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/
│   │   ├── test_kernels.cu        # SPH kernel correctness
│   │   ├── test_lbvh.cu           # LBVH neighbor query correctness
│   │   ├── test_particle_store.cu
│   │   └── test_scene_loader.cpp
│   └── regression/
│       ├── test_dfsph_dam_break.cu
│       └── golden/                # frozen reference outputs
├── scenes/
│   ├── glass_pour.json
│   ├── dam_break.json             # regression scene
│   └── single_drop.json           # smallest debug scene
├── assets/
│   ├── hdri/        # gitignored, large
│   ├── meshes/
│   └── textures/
├── third_party/
│   └── CMakeLists.txt
├── legacy/                        # ALL of the original 2023 code, untouched
│   ├── src/
│   ├── include/
│   ├── shaders/
│   └── ... (full original tree)
└── .gitignore
```

---

## 10. Memory budget (8 GB VRAM)

Sized for 800K fluid particles + 384³ density grid + a 3M-triangle BLAS + path tracer working set.

| Allocation | Size |
|---|---|
| Driver / OS / Vulkan reserved | ~1.0 GB |
| Particle store (800K × ~200 bytes/particle) | 0.16 GB |
| LBVH (nodes + AABBs) | 0.05 GB |
| SDF boundary 256³ | 0.07 GB |
| Anisotropy matrices (800K × 36 bytes) | 0.03 GB |
| Density grid (384³ × 4 bytes) | 0.23 GB |
| MC vertex buffer (3M tris × 36 bytes) | 0.11 GB |
| MC index buffer (3M tris × 12 bytes) | 0.04 GB |
| BLAS storage | 0.30 GB |
| TLAS + scratch | 0.05 GB |
| Path tracer accum buffer (1080p × 16 bytes RGBA32F) | 0.03 GB |
| Aux buffers (albedo, normal, depth, motion) | 0.10 GB |
| Per-pixel path state (1080p × 64 bytes) | 0.13 GB |
| OIDN working memory | ~0.5 GB |
| HDRI + textures | ~0.5 GB |
| Shader binaries + descriptors | ~0.05 GB |
| Headroom | ~4.6 GB |

**Result:** ~3.4 GB used → comfortable on 8 GB. Headroom lets us experiment with higher particle counts or grid resolution.

**Risk:** if MC mesh exceeds 5M triangles regularly, BLAS storage grows; we mitigate by reducing MC grid resolution adaptively.

---

## 11. Testing strategy

### 11.1 Unit tests (doctest)

- `test_kernels.cu` — SPH kernel values at known points; integral ≈ 1
- `test_lbvh.cu` — neighbor query returns same set as O(N²) brute force on small inputs
- `test_particle_store.cu` — alloc, view, resize, attribute registration
- `test_scene_loader.cpp` — JSON parse, validation errors, asset resolution
- `test_sdf_sampler.cu` — distance to known shapes (sphere, plane)
- `test_anisotropy.cu` — SVD of known matrices

### 11.2 Regression tests

**Dam break** (`tests/regression/test_dfsph_dam_break.cu`):
- Classic 2D dam-break scene, simulate 100 frames
- Compare frame-100 particle positions against golden reference (committed in `tests/regression/golden/`)
- Pass if mean particle position deviation < 1% of domain size
- Updated by hand if solver semantics intentionally change (e.g., bug fix)

### 11.3 Visual regression

For renderer changes — render `scenes/single_drop.json` at 64 SPP, compare against committed golden PNG via SSIM threshold. Cheap; runs in CI.

### 11.4 Smoke test in CI

A 10-particle, 5-frame, 16-SPP run that exercises every module. Catches "did everything compile and link" without burning GPU minutes.

---

## 12. Milestones

Working backward from "the v1 cinematic exists":

**Phase 1: Foundation (Week 1-2)**
- M1.1: Modern CMake build, all dependencies fetched
- M1.2: Particle store + LBVH + SDF boundary core
- M1.3: Vulkan device init + Slang toolchain compiling hello-compute
- M1.4: Scene loader for trivial JSON scene
- M1.5: Tests passing, smoke test in CI
- ✅ **Exit criterion:** `./sim_cli --scene single_drop.json` runs and exits cleanly with no rendering yet

**Phase 2: Solver (Week 3-4)**
- M2.1: DFSPH density solver — converges on dam-break
- M2.2: DFSPH divergence solver
- M2.3: Surface tension (Akinci) integrated
- M2.4: Adaptive timestep + checkpointing
- M2.5: Regression test passes
- ✅ **Exit criterion:** dam-break scene runs to completion, particle state matches golden within tolerance

**Phase 3: Viewport (Week 5)**
- M3.1: Screen-space fluid renderer
- M3.2: Camera controls + ImGui overlay
- M3.3: Hot-reload of scene parameters
- ✅ **Exit criterion:** can see fluid simulating in real-time, tune parameters live

**Phase 4: Surface reconstruction (Week 6-7)**
- M4.1: Anisotropy matrix computation per particle
- M4.2: Density grid sampling
- M4.3: Marching Cubes mesh extraction
- M4.4: Mesh → Vulkan-imported buffer
- ✅ **Exit criterion:** triangle mesh of fluid surface visible in a debug Vulkan rasterizer

**Phase 5: Path tracer (Week 8-10)**
- M5.1: BLAS/TLAS construction from mesh
- M5.2: Ray gen + miss + simple Lambertian closest-hit
- M5.3: BSDF library (rough dielectric, Disney diffuse)
- M5.4: Environment HDRI sampling
- M5.5: ReSTIR DI for direct lighting
- M5.6: Camera with DoF + motion blur
- M5.7: Frame accumulation + tonemap
- ✅ **Exit criterion:** can render a glass-pour frame at 256 SPP

**Phase 6: Final cinematic (Week 11-12)**
- M6.1: OIDN denoise integration
- M6.2: EXR sequence + ffmpeg mux
- M6.3: Render the final 120 frames
- M6.4: Polish, color grade, output `glass_pour_v1.mp4`
- ✅ **Exit criterion:** the video file exists, looks cinematic

**Reality:** these are calendar weeks of focused work, not Claude-assisted hours. With Claude in the loop, expect ~2× speedup, so ~6-8 calendar weeks.

---

## 13. Risks & open questions

### 13.1 Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Vulkan-RT learning curve eats Phase 5 | High | High | Start renderer scaffolding early in Phase 1, even if just hello-triangle |
| Anisotropic kernel SVD numerical issues | Medium | Medium | Reference SPlisHSPlasH impl; add eigenvalue regularization |
| Caustics need BDPT, not just ReSTIR | Medium | High | If unidirectional + ReSTIR can't resolve caustics in 2048 SPP, fall back to manifold-NEE or photon mapping in v2 |
| 8 GB VRAM tight when MC mesh is large | Medium | Medium | Adaptive MC resolution; consider narrow-band level set if needed |
| Slang tooling immaturity (debug, error msgs) | Low | Medium | Slang has matured significantly; fallback is HLSL-compiled-to-SPIRV |
| CUDA-Vulkan interop fiddly on Linux | Medium | Medium | Use existing samples as references (NVIDIA's `cuda_samples/cudaVulkanInterop`) |

### 13.2 Open questions

- **OIDN backend:** CPU vs CUDA? CUDA backend keeps data on-GPU but adds dependency complexity. Default to CUDA backend; fall back to CPU if blocking.
- **Glass mesh source:** model in Blender or use a procedural primitive (cylinder + difference)? **Defer:** start procedural for first renders, model in Blender for hero shot.
- **HDRI source:** HDRI Haven (Poly Haven) free assets are perfect. License: CC0. **Decided.**
- **CI:** GitHub Actions GPU runners are expensive. **Decided:** self-hosted runner on user's own machine, or skip GPU CI entirely and rely on smoke tests in pre-commit.

---

## 14. v2+ roadmap

Documented here so module boundaries can anticipate them, even though they're out of scope for v1.

**v2 — Splash shot:**
- FLIP solver added behind `Solver` interface (shares ParticleStore, LBVH, Boundary, Renderer)
- Sphere-drops-into-pool scene
- Foam/spray particle system (separate from fluid particles)
- Direct implicit ray tracing of anisotropic surface (no MC mesh) — procedural primitives in Vulkan RT BLAS
- BDPT or photon mapping for hard caustics

**v3 — Multi-fluid:**
- Two-phase fluids (water + air with drag coupling)
- Colored / dyed liquids (volumetric absorption)
- Subsurface scattering for milky liquids
- Wet-glass exterior droplets

**v∞:**
- MPM solver for snow, sand, viscoelastic fluids
- Differentiable rendering via Slang auto-diff (fit sim params to reference video)
- Cross-vendor support (Vulkan portability, AMD ROCm/HIP backend)

---

## 15. References

**Solver:**
- Bender & Koschier, *Divergence-Free Smoothed Particle Hydrodynamics*, SCA 2015
- Akinci et al., *Versatile Surface Tension and Adhesion for SPH Fluids*, SIGGRAPH Asia 2013
- Akinci et al., *Versatile Rigid-Fluid Coupling for Incompressible SPH*, SIGGRAPH 2012
- Ihmsen et al., *Implicit Incompressible SPH*, IEEE TVCG 2014
- SPlisHSPlasH reference implementation: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

**Spatial acceleration:**
- Karras, *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*, HPG 2012
- Aila & Laine, *Understanding the Efficiency of Ray Traversal on GPUs*, HPG 2009

**Surface reconstruction:**
- Yu & Turk, *Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels*, ACM TOG 2013
- Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987

**Rendering:**
- PBRT 4th edition (Pharr/Jakob/Humphreys)
- Bitterli et al., *Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting*, SIGGRAPH 2020
- NVIDIA Vulkan Ray Tracing Tutorial: https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
- Slang: https://shader-slang.com/

**Viewport:**
- Müller, *Screen Space Meshes*, SCA 2007
- van der Laan, *Screen Space Fluid Rendering with Curvature Flow*, I3D 2009

---

*End of spec.*

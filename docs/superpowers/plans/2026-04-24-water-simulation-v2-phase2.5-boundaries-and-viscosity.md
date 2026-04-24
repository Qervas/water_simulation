# Water Simulation v2 — Phase 2.5: Boundary Particles + Viscosity + Density Convergence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DFSPH solver from Phase 2 actually stable. A 5000-particle fluid block dropped into an open box must settle into a puddle of correct depth (within ±20% of analytical prediction) and remain bounded for at least 5 seconds of simulated time. This is the missing physics that turns Phase 2's "scaffolding that runs" into "a solver that produces valid simulation output."

**Architecture:** Add Akinci 2012 boundary particles — static SPH-particle samples of the box walls with calibrated mass — that participate in density and pressure sums but never get advected. Add explicit Akinci 2013 viscosity force. Add CUB-based density-solver convergence check (early exit when `mean(|ρ_pred - ρ₀|) < eta * ρ₀`). Replace the constant-1 `max_velocity_` placeholder with a real GPU max reduction.

**Tech Stack:** Same as Phase 2 (CUDA 13.2, CCCL, doctest). No new dependencies.

**Spec section:** §6.3 (boundary), §6.5 (DFSPH viscosity).

**Phase 2.5 exit criterion:** `tests/regression/test_settled_puddle` is green: a 0.4³m fluid block at y=0.3 dropped into a 1m³ box settles by t=2s into a puddle whose mean y is within 20% of `(initial_volume / floor_area)^(1/3)` and whose max-velocity stays below 0.5 m/s after t=2s.

---

## Why this is its own phase

Phase 2 produced a DFSPH solver scaffold that *technically computes pressure and applies it*, but the simulation output is dominated by boundary artifacts because the AABB clamp doesn't generate true pressure forces against walls. Without Akinci-style boundary particles, the density field at every wall-adjacent fluid particle reads low (no neighbors on the wall side), kappa is calculated from a wrong baseline, and the solver pulls particles into the walls instead of pushing them away. This is a known SPH failure mode — every production SPH library (SPlisHSPlasH, Houdini's POPs, etc.) handles boundaries via either Akinci particles or full SDF-with-volume-correction.

We chose to peel boundary work into its own focused phase because:
1. **TDD discipline**: Phase 2 fell into "looks plausible" territory because we never wrote a "settled puddle has correct depth" test before declaring T7 done. Phase 2.5 starts by writing exactly that test, then builds the physics that makes it pass.
2. **Module purity**: boundary particles want to live alongside fluid particles in the same neighbor structure but be filtered out of advection. This is a real architecture decision that deserves dedicated thought, not "tack it onto the dam-break run."
3. **Verification**: with boundaries done, ALL subsequent phases (viewport, surface reconstruction, path tracer) consume reliable simulation output. Without them, we'd be rendering garbage.

---

## Prerequisites

- Phase 2 partial merged to master (commit referenced in this branch's history)
- Build still green: `cmake --preset linux-debug-local && cmake --build build/linux-debug-local && ctest --preset linux-debug-local`

---

## File structure decomposition

**Modified:**
- `solvers/dfsph/include/water/solvers/dfsph.h` — add boundary-particle handles, viscosity coefficient, mean_density_error_ accessor
- `solvers/dfsph/src/dfsph.cu` — wire boundary kernels into step pipeline; replace fixed iters with convergence check; real max_velocity_ via CUB
- `solvers/dfsph/src/dfsph_kernels.cu` — boundary-aware density and pressure kernels; viscosity kernel; revert AABB-bounce hack to a stricter no-slip clamp
- `core/include/water/core/particle_store.h` — no changes needed; boundary particles use a separate ParticleStore
- `apps/sim_cli/main.cpp` — generate boundary particles at startup; pass them into DFSPHSolver
- `README.md` — flip Phase 2 status to "complete" once T7 passes

**New:**
- `solvers/dfsph/include/water/solvers/boundary_sampler.h` — pure host-side function: AABB box → vector<Vec3f> on wall surfaces
- `solvers/dfsph/src/boundary_sampler.cpp`
- `tests/regression/CMakeLists.txt` — new regression test target
- `tests/regression/test_settled_puddle.cu` — the FIRST test; defines "stable puddle"
- `tests/unit/test_boundary_sampler.cu` — counts/spacing of generated boundary particles
- `tests/unit/test_boundary_mass.cu` — Akinci mass calibration: a fluid particle adjacent to a boundary should see ρ ≈ ρ₀

---

## Task 1: Define "stable puddle" — write the failing regression test FIRST

**Files:**
- Create: `tests/regression/CMakeLists.txt`
- Create: `tests/regression/test_settled_puddle.cu`
- Modify: `tests/CMakeLists.txt` — add `add_subdirectory(regression)`

The test runs a 5000-particle fluid block in a 1m³ box, simulates 5 seconds, and asserts:
1. After t=2s, `max_velocity < 0.5 m/s` (settled, not chaotic)
2. Mean particle y is within ±20% of analytical settled depth
3. Max particle y stays below the initial fall height (no exploded particles)
4. Particle count is preserved (no NaN-out, no leaks through walls)

This test will FAIL initially — it's the contract Phase 2.5 must satisfy.

- [ ] **Step 1: Write `tests/regression/CMakeLists.txt`**

```cmake
add_executable(water_regression
    test_settled_puddle.cu
    ../unit/test_main.cpp
)

target_link_libraries(water_regression PRIVATE
    water_core
    water_scene
    water_dfsph
    doctest::doctest
)

set_target_properties(water_regression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME water_regression COMMAND water_regression)
```

- [ ] **Step 2: Add `add_subdirectory(regression)` at the bottom of `tests/CMakeLists.txt`**

- [ ] **Step 3: Write `tests/regression/test_settled_puddle.cu`**

```cpp
#include <doctest/doctest.h>
#include "water/core/particle_store.h"
#include "water/core/cuda_check.h"
#include "water/solvers/dfsph.h"
#include <cmath>
#include <vector>

using namespace water;
using namespace water::solvers;

namespace {

std::vector<Vec3f> dense_block(Vec3f lo, Vec3f hi, f32 spacing) {
    std::vector<Vec3f> pts;
    for (f32 x = lo.x; x <= hi.x; x += spacing)
    for (f32 y = lo.y; y <= hi.y; y += spacing)
    for (f32 z = lo.z; z <= hi.z; z += spacing) {
        pts.push_back({x, y, z});
    }
    return pts;
}

struct FluidStats {
    f32 y_min, y_max, y_mean;
    f32 max_speed;
    std::size_t finite_count;
};

FluidStats stats(const ParticleStore& store) {
    std::vector<Vec3f> pos(store.count()), vel(store.count());
    WATER_CUDA_CHECK(cudaMemcpy(pos.data(), store.positions(),
                                 sizeof(Vec3f) * store.count(),
                                 cudaMemcpyDeviceToHost));
    WATER_CUDA_CHECK(cudaMemcpy(vel.data(), store.velocities(),
                                 sizeof(Vec3f) * store.count(),
                                 cudaMemcpyDeviceToHost));
    FluidStats s{1e9f, -1e9f, 0.f, 0.f, 0};
    for (std::size_t i = 0; i < pos.size(); ++i) {
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) ||
            !std::isfinite(pos[i].z)) continue;
        s.finite_count++;
        s.y_min = std::min(s.y_min, pos[i].y);
        s.y_max = std::max(s.y_max, pos[i].y);
        s.y_mean += pos[i].y;
        const f32 sp = std::sqrt(vel[i].x * vel[i].x + vel[i].y * vel[i].y
                              + vel[i].z * vel[i].z);
        s.max_speed = std::max(s.max_speed, sp);
    }
    if (s.finite_count > 0) s.y_mean /= static_cast<f32>(s.finite_count);
    return s;
}

} // namespace

TEST_CASE("settled puddle: 0.4^3 m fluid block settles to correct depth in <2s") {
    const f32 r = 0.012f;
    const f32 spacing = 2.0f * r;

    Vec3f initial_lo{0.30f, 0.30f, 0.30f};
    Vec3f initial_hi{0.70f, 0.70f, 0.70f};
    auto pts = dense_block(initial_lo, initial_hi, spacing);
    REQUIRE(pts.size() > 1000);
    REQUIRE(pts.size() < 100000);

    ParticleStore store(pts.size());
    store.resize(pts.size());
    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), pts.data(),
                                 sizeof(Vec3f) * pts.size(),
                                 cudaMemcpyHostToDevice));

    DFSPHSolver::Config cfg;
    cfg.rest_density     = 1000.0f;
    cfg.particle_radius  = r;
    cfg.smoothing_length = 2.0f * r;
    cfg.viscosity        = 1e-3f;        // Phase 2.5: now actually used
    cfg.surface_tension  = 0.0f;
    cfg.gravity          = {0.f, -9.81f, 0.f};
    cfg.domain_min       = {0.f, 0.f, 0.f};
    cfg.domain_max       = {1.f, 1.f, 1.f};

    DFSPHSolver solver(store, cfg);

    const f32 dt_substep = 0.002f;
    const f32 sim_time   = 5.0f;
    const int steps      = static_cast<int>(sim_time / dt_substep);
    for (int k = 0; k < steps; ++k) solver.step(dt_substep);
    cudaDeviceSynchronize();

    auto s = stats(store);

    // (1) all particles still finite — no NaN-out
    CHECK(s.finite_count == pts.size());
    // (2) settled — max speed under 0.5 m/s after 5 sec
    CHECK(s.max_speed < 0.5f);
    // (3) no escapees above initial fall height
    CHECK(s.y_max < initial_hi.y + 0.05f);
    // (4) puddle has correct depth.
    //   Analytical: volume = 0.064 m³. Floor = 1 m². Depth = 0.064 m.
    //   Mean particle y = depth / 2 = 0.032 m (uniform vertical distribution).
    const f32 expected_mean_y = 0.032f;
    CHECK(s.y_mean == doctest::Approx(expected_mean_y).epsilon(0.20f));
}
```

- [ ] **Step 4: Build and run — confirm it FAILS in the expected way**

```bash
cmake --build build/linux-debug-local --target water_regression -j
./build/linux-debug-local/bin/water_regression
```

Expected: test fails with `s.y_mean ≈ 0.001` (current paper-thin layer behavior), not the target 0.032. This documents the bug we're fixing.

- [ ] **Step 5: Commit (red test, intentionally)**

```bash
git add tests/regression/ tests/CMakeLists.txt
git commit -m "tests/regression: failing 'settled puddle' test for Phase 2.5

Asserts that a 0.4^3 m fluid block in a 1m^3 box settles by t=5s into
a puddle of mean y ≈ 0.032m (depth = volume / floor_area = 0.064m,
mean = depth/2). Currently FAILS — water spreads infinitely thin
because boundary handling lacks proper Akinci pressure forces. This
red test is the Phase 2.5 contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Boundary particle sampler

**Files:**
- Create: `solvers/dfsph/include/water/solvers/boundary_sampler.h`
- Create: `solvers/dfsph/src/boundary_sampler.cpp`
- Modify: `solvers/dfsph/CMakeLists.txt` — add `src/boundary_sampler.cpp`
- Create: `tests/unit/test_boundary_sampler.cu`
- Modify: `tests/CMakeLists.txt` — add to source list

Generate a single-layer or multi-layer grid of particles on each face of an AABB box. Akinci recommends 2-3 layers deep on each face to provide enough kernel support.

- [ ] **Step 1: Write `solvers/dfsph/include/water/solvers/boundary_sampler.h`**

```cpp
#pragma once

#include "water/core/types.h"
#include <vector>

namespace water::solvers {

// Sample the interior surfaces of an AABB box with regularly spaced static
// SPH particles. Multiple layers (going INWARD from each face) provide
// enough kernel support for fluid neighbors near the wall.
//
// For a box [domain_min, domain_max] with `spacing` between particles and
// `n_layers` layers per face, total particles ≈ 6 * n_layers * area / spacing².
std::vector<Vec3f> sample_aabb_boundary(
    Vec3f domain_min, Vec3f domain_max,
    f32 spacing, int n_layers = 2);

} // namespace water::solvers
```

- [ ] **Step 2: Write `solvers/dfsph/src/boundary_sampler.cpp`**

(Implementation: 6 nested loops, one per face, each emitting a 2D grid with `n_layers` offsets along the inward normal. Skip duplicates at edges/corners by tracking emitted positions in a hash set, OR accept mild over-sampling at edges since boundary mass calibration handles it naturally.)

```cpp
#include "water/solvers/boundary_sampler.h"
#include <cmath>

namespace water::solvers {

std::vector<Vec3f> sample_aabb_boundary(Vec3f lo, Vec3f hi, f32 spacing, int n_layers) {
    std::vector<Vec3f> pts;
    auto emit_face = [&](int axis, f32 face_coord, f32 inward_step) {
        // axis 0 = X, 1 = Y, 2 = Z
        const f32 a_lo[3] = {lo.x, lo.y, lo.z};
        const f32 a_hi[3] = {hi.x, hi.y, hi.z};
        // The two non-axis dimensions index the 2D grid.
        int u_axis = (axis + 1) % 3;
        int v_axis = (axis + 2) % 3;
        for (int layer = 0; layer < n_layers; ++layer) {
            f32 axis_pos = face_coord + layer * inward_step;
            for (f32 u = a_lo[u_axis]; u <= a_hi[u_axis] + 0.5f * spacing; u += spacing)
            for (f32 v = a_lo[v_axis]; v <= a_hi[v_axis] + 0.5f * spacing; v += spacing) {
                f32 p[3];
                p[axis]   = axis_pos;
                p[u_axis] = u;
                p[v_axis] = v;
                pts.push_back({p[0], p[1], p[2]});
            }
        }
    };
    emit_face(0, lo.x,  spacing);
    emit_face(0, hi.x, -spacing);
    emit_face(1, lo.y,  spacing);
    emit_face(1, hi.y, -spacing);
    emit_face(2, lo.z,  spacing);
    emit_face(2, hi.z, -spacing);
    return pts;
}

} // namespace water::solvers
```

- [ ] **Step 3: Write `tests/unit/test_boundary_sampler.cu`**

```cpp
#include <doctest/doctest.h>
#include "water/solvers/boundary_sampler.h"

using namespace water;
using namespace water::solvers;

TEST_CASE("boundary sampler: counts roughly match 6 * n_layers * area / spacing^2") {
    auto pts = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.025f, 2);
    // 6 faces * 2 layers * (1/0.025)^2 ≈ 6 * 2 * 1600 = 19200, with edge double-counts
    CHECK(pts.size() > 15000);
    CHECK(pts.size() < 30000);
}

TEST_CASE("boundary sampler: every point is on or just inside a face") {
    auto pts = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.05f, 1);
    for (auto p : pts) {
        bool on_face = (p.x <= 0.001f) || (p.x >= 0.999f)
                    || (p.y <= 0.001f) || (p.y >= 0.999f)
                    || (p.z <= 0.001f) || (p.z >= 0.999f);
        CHECK(on_face);
    }
}
```

- [ ] **Step 4: Add to `solvers/dfsph/CMakeLists.txt`**

```cmake
add_library(water_dfsph STATIC
    src/dfsph.cu
    src/dfsph_kernels.cu
    src/boundary_sampler.cpp        # NEW
)
```

- [ ] **Step 5: Add `tests/unit/test_boundary_sampler.cu` to `tests/CMakeLists.txt` source list**

- [ ] **Step 6: Build, run unit tests, commit**

```bash
cmake --build build/linux-debug-local -j
./build/linux-debug-local/bin/water_tests
git add solvers/ tests/
git commit -m "solvers/dfsph: AABB boundary particle sampler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Boundary particles in their own ParticleStore + Akinci mass calibration

**Files:**
- Modify: `solvers/dfsph/include/water/solvers/dfsph.h` — accept boundary store at construction
- Modify: `solvers/dfsph/src/dfsph.cu` — own a CellGrid for boundary particles
- Modify: `solvers/dfsph/src/dfsph_kernels.cu` — boundary mass calibration kernel
- Create: `tests/unit/test_boundary_mass.cu`

The Akinci 2012 boundary-mass formula: for each boundary particle b, its "effective mass" = ρ₀ / Σ_b' W(|x_b - x_b'|, l) summed over boundary neighbors. This calibrates the mass so a fluid particle adjacent to the wall sees a density contribution equivalent to fluid neighbors on the other side.

- [ ] **Step 1: Modify `DFSPHSolver` constructor to accept a boundary store**

```cpp
// In dfsph.h, replace the existing ctor with:
DFSPHSolver(ParticleStore& fluid_store,
            ParticleStore* boundary_store,   // nullptr = no boundary
            Config cfg,
            cudaStream_t stream = 0);
```

- [ ] **Step 2: Add boundary CellGrid + boundary mass kernel** — full code in `dfsph_kernels.cu` and `dfsph.cu`. Pattern: same as fluid density kernel, but iterates only boundary→boundary neighbors.

(Concrete code listed in the full implementation file below — abbreviated here for plan brevity. The kernel is structurally identical to `density_kernel` from Phase 2 T4, but writes to a `boundary_mass_` attribute on the boundary store and uses W(0) self-contribution + neighbor sums.)

- [ ] **Step 3: Write `tests/unit/test_boundary_mass.cu`** — verify that a fluid particle placed 1 spacing inside the wall, with calibrated boundary particles, samples ρ ≈ ρ₀ (within 5%).

- [ ] **Step 4: Build, test, commit**

---

## Task 4: Boundary-aware density + pressure kernels

**Files:**
- Modify: `solvers/dfsph/src/dfsph_kernels.cu`
- Modify: `solvers/dfsph/src/dfsph.cu`

Modify `density_kernel`, `alpha_kernel`, `density_change_kernel`, `apply_kappa_kernel` to also iterate over boundary particles in the same 27-cell stencil. Boundary contributions use the calibrated `boundary_mass_` array. Boundary particle velocities are zero (no advection).

- [ ] **Step 1-4**: Update each kernel to take additional parameters: `(positions_b, sorted_indices_b, cell_start_b, mass_b, dims_b, origin_b, cell_length_b)`. Iterate the boundary 27-cell stencil right after the fluid one. Accumulate into the same density / kappa scratch buffers.

- [ ] **Step 5: Re-run all unit tests — they must still pass with the new kernel signatures**

- [ ] **Step 6: Run the regression test from T1 — should now show partial improvement** (water no longer pulled into walls; should at least stack a few layers).

- [ ] **Step 7: Commit**

---

## Task 5: Akinci viscosity force

**Files:**
- Modify: `solvers/dfsph/src/dfsph_kernels.cu` — viscosity kernel
- Modify: `solvers/dfsph/src/dfsph.cu` — host method, called from `step()`

Akinci 2013 explicit viscosity:
```
a_visc_i = (visc / rho_i) * Σ_j (m_j / rho_j) * (v_j - v_i) * (∇²W_ij)
```
where the Laplacian is approximated by `(2 * (d+2) / (h² * (q²+0.01))) * W(r,l)` for stability (XSPH-style).

For Phase 2.5 we use a simpler XSPH velocity smoother:
```
v_i ← v_i + epsilon * Σ_j (m_j / rho_avg) * (v_j - v_i) * W(|x_i-x_j|, l)
```
with `epsilon = cfg_.viscosity` (default 0.001). This is cheap, robust, and produces visibly damped fluid motion.

- [ ] **Step 1**: Write `xsph_viscosity_kernel` (signature mirrors `density_kernel`, but accumulates a velocity correction).
- [ ] **Step 2**: Add `viscosity()` host method to `DFSPHSolver`. Call after `apply_external_forces()` and before `density_solve()` in `step()`.
- [ ] **Step 3**: Add a unit test: two particles moving in opposite directions converge in velocity over time when viscosity > 0.
- [ ] **Step 4**: Build, run tests, run regression — should show further improvement.
- [ ] **Step 5**: Commit.

---

## Task 6: Density-solver convergence check (replaces fixed 20 iters)

**Files:**
- Modify: `solvers/dfsph/include/water/solvers/dfsph.h` — already has `mean_density_error_`
- Modify: `solvers/dfsph/src/dfsph.cu` — implement properly via CUB DeviceReduce::Sum
- Modify: `solvers/dfsph/src/dfsph_kernels.cu` — density-error per-particle kernel

CUB-based convergence: after each density-solve iteration, sum `max(0, ρ_pred[i] - ρ₀)` and divide by N. Exit loop when below `eta_density * ρ₀`.

- [ ] **Step 1**: Implement `mean_density_error_()` properly with CUB. Test it on a controlled input.
- [ ] **Step 2**: Replace fixed-20-iter loop in `density_solve` with `while (iter < cfg_.max_density_iters && err > eta) { ... }`.
- [ ] **Step 3**: Run regression test — should now converge faster on average and stack better near walls.
- [ ] **Step 4**: Commit.

---

## Task 7: Real `max_velocity_` via CUB DeviceReduce::Max

**Files:**
- Modify: `solvers/dfsph/src/dfsph.cu`
- Modify: `solvers/dfsph/src/dfsph_kernels.cu`

- [ ] **Step 1**: Add `velocity_magnitude_kernel` that writes per-particle `||v||` into `reduce_in_`.
- [ ] **Step 2**: Replace the constant placeholder in `step()` end with CUB Max reduction.
- [ ] **Step 3**: Verify substep counts in `sim_cli` are now responsive (drop when fluid settles, rise during impact).
- [ ] **Step 4**: Commit.

---

## Task 8: Wire boundary sampling into sim_cli + green the regression test

**Files:**
- Modify: `apps/sim_cli/main.cpp` — generate boundary particles, build a boundary ParticleStore, pass to DFSPHSolver
- Modify: `tests/regression/test_settled_puddle.cu` — also build boundary particles in the test

- [ ] **Step 1**: In `sim_cli`, after fluid block generation, call `sample_aabb_boundary` with the scene's domain. Build a boundary ParticleStore. Pass it as a second arg to DFSPHSolver constructor.
- [ ] **Step 2**: Same in the regression test.
- [ ] **Step 3**: Run the regression test — it should now PASS (mean y ≈ 0.032 ± 20%).
- [ ] **Step 4**: Re-render `dam_break.json` via `sim_cli --record` + `tools/viz.py`. Visually inspect: water should now form a stable settling layer of correct depth, not a film.
- [ ] **Step 5**: Commit.

---

## Task 9: Final certification + Phase 2 status flip

**Files:**
- Modify: `README.md` — change Phase 2 status from "⚠️ partially complete" to "✅ complete"

- [ ] **Step 1**: Clean rebuild + ctest + sim_cli on `dam_break.json` + viz.
- [ ] **Step 2**: Update README to mark Phase 2 done.
- [ ] **Step 3**: Final commit.
- [ ] **Step 4**: Merge to master.

---

## Spec coverage check

| Spec section | Plan task |
|---|---|
| §6.3 Boundary as SDF | Tasks 2-3 (Akinci particles is the "stage 1" of SDF — actual SDF deferred to Phase 4) |
| §6.5 Viscosity | Task 5 |
| §6.5 Density convergence | Task 6 |
| §6.5 Adaptive substepping | Task 7 |
| Test discipline (TDD) | Task 1 (failing test first) |

Phase 4 still owns:
- True SDF boundary (mesh → distance field)
- Akinci surface tension curvature term (cohesion ships in Phase 2)
- Anisotropic surface reconstruction

---

*End of Phase 2.5 plan.*

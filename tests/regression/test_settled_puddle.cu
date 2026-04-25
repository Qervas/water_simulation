// tests/regression/test_settled_puddle.cu
//
// Phase 2.5 contract: a 0.4^3 m fluid block dropped in a 1m^3 box must settle
// by t=5s into a puddle of correct depth and stay bounded.
//
// This test starts FAILING. Phase 2.5 (boundary particles + viscosity +
// density-solve convergence) is the work that makes it pass.

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

FluidStats compute_stats(const ParticleStore& store) {
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

TEST_CASE("settled puddle: 0.4^3 m block settles into a puddle of correct depth") {
    const f32 r = 0.012f;
    const f32 spacing = 2.0f * r;

    const Vec3f initial_lo{0.30f, 0.30f, 0.30f};
    const Vec3f initial_hi{0.70f, 0.70f, 0.70f};
    auto pts = dense_block(initial_lo, initial_hi, spacing);
    REQUIRE(pts.size() > 1000);
    REQUIRE(pts.size() < 100000);
    INFO("particle count = " << pts.size());

    ParticleStore store(pts.size());
    store.resize(pts.size());
    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), pts.data(),
                                 sizeof(Vec3f) * pts.size(),
                                 cudaMemcpyHostToDevice));

    DFSPHSolver::Config cfg;
    cfg.rest_density     = 1000.0f;
    cfg.particle_radius  = r;
    cfg.smoothing_length = 2.0f * r;
    cfg.viscosity        = 1e-3f;
    cfg.surface_tension  = 0.0f;
    cfg.gravity          = {0.f, -9.81f, 0.f};
    cfg.domain_min       = {0.f, 0.f, 0.f};
    cfg.domain_max       = {1.f, 1.f, 1.f};

    DFSPHSolver solver(store, cfg);

    // 5 seconds simulated time at dt=2ms = 2500 substeps.
    const f32 dt = 0.002f;
    const int steps = 2500;
    for (int k = 0; k < steps; ++k) solver.step(dt);
    cudaDeviceSynchronize();

    auto s = compute_stats(store);

    INFO("y_min=" << s.y_min << " y_max=" << s.y_max
         << " y_mean=" << s.y_mean << " max_speed=" << s.max_speed
         << " finite=" << s.finite_count);

    // (1) all particles still finite — no NaN-out
    CHECK(s.finite_count == pts.size());

    // (2) settled — max speed under 0.5 m/s after 5 sec
    CHECK(s.max_speed < 0.5f);

    // (3) no escapees above initial fall height
    CHECK(s.y_max < initial_hi.y + 0.05f);

    // (4) puddle depth correct.
    //   Particle volume per particle = (2r)^3 = 1.38e-5 m^3.
    //   Total volume = N * 1.38e-5 m^3.
    //   Floor area = 1 m^2 (clamped to domain).
    //   Expected puddle depth ≈ N * 1.38e-5 m.
    //   For ~9000 particles this gives ~0.124 m depth.
    //   Mean particle y in a uniformly-filled puddle = depth / 2.
    const f32 particle_volume = (2.0f * r) * (2.0f * r) * (2.0f * r);
    const f32 floor_area      = 1.0f;
    const f32 expected_depth  = pts.size() * particle_volume / floor_area;
    const f32 expected_mean_y = expected_depth / 2.0f;
    INFO("expected_depth=" << expected_depth
         << " expected_mean_y=" << expected_mean_y);
    CHECK(s.y_mean == doctest::Approx(expected_mean_y).epsilon(0.30f));
}

#include <doctest/doctest.h>
#include "water/solvers/dfsph.h"
#include "water/core/sph_kernels.cuh"
#include "water/core/cuda_check.h"
#include <vector>

using namespace water;
using namespace water::solvers;

namespace {
f32 host_W0(f32 l) {
    return 1.0f / (3.14159265358979323846f * l * l * l);
}
} // namespace

TEST_CASE("DFSPHSolver: density of a single isolated particle ≈ mass * W(0)") {
    ParticleStore store(1);
    store.resize(1);
    Vec3f p{0.5f, 0.5f, 0.5f};
    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), &p, sizeof(Vec3f),
                                 cudaMemcpyHostToDevice));

    DFSPHSolver::Config cfg;
    cfg.rest_density     = 1000.0f;
    cfg.particle_radius  = 0.01f;
    cfg.smoothing_length = 0.04f;
    cfg.domain_min       = {0.f, 0.f, 0.f};
    cfg.domain_max       = {1.f, 1.f, 1.f};

    DFSPHSolver solver(store, cfg);
    solver.step(0.001f);
    cudaDeviceSynchronize();

    const f32 expected = solver.particle_mass() * host_W0(cfg.smoothing_length);
    CHECK(solver.density_at(0) == doctest::Approx(expected).epsilon(1e-4f));
}

TEST_CASE("DFSPHSolver: dense pack center density approaches rest_density") {
    const f32 r = 0.01f;
    const f32 spacing = 2.0f * r;
    std::vector<Vec3f> pts;
    for (int z = 0; z < 10; ++z)
    for (int y = 0; y < 10; ++y)
    for (int x = 0; x < 10; ++x) {
        pts.push_back({0.1f + x * spacing, 0.1f + y * spacing, 0.1f + z * spacing});
    }

    ParticleStore store(pts.size());
    store.resize(pts.size());
    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), pts.data(),
                                 sizeof(Vec3f) * pts.size(),
                                 cudaMemcpyHostToDevice));

    DFSPHSolver::Config cfg;
    cfg.rest_density     = 1000.0f;
    cfg.particle_radius  = r;
    cfg.smoothing_length = 4.0f * r;
    cfg.domain_min       = {0.f, 0.f, 0.f};
    cfg.domain_max       = {1.f, 1.f, 1.f};

    DFSPHSolver solver(store, cfg);
    solver.step(0.001f);
    cudaDeviceSynchronize();

    // Center particle: index = 5*100 + 5*10 + 5 = 555
    const f32 d = solver.density_at(555);
    // Pre-solver baseline check: density should be in the right ballpark
    // (within 30%) — SPH discrete-sampling error at this neighbor count is
    // expected. The iterative density-invariance solver in Task 7 corrects
    // residual error to within eta_density (typ. 0.1%).
    CHECK(d == doctest::Approx(cfg.rest_density).epsilon(0.30f));
    CHECK(d > 500.0f);
    CHECK(d < 1500.0f);
}

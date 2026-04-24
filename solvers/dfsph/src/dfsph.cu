#include "water/solvers/dfsph.h"
#include "water/core/cuda_check.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace water::solvers {

namespace detail {
__global__ void density_kernel(
    const Vec3f*, const u32*, const u32*, f32*,
    std::size_t, Vec3f, Vec3i, f32, f32, f32);
__global__ void alpha_kernel(
    const Vec3f*, const f32*, const u32*, const u32*, f32*,
    std::size_t, Vec3f, Vec3i, f32, f32, f32);
__global__ void apply_gravity_kernel(Vec3f*, std::size_t, Vec3f, f32);
__global__ void advect_kernel(Vec3f*, Vec3f*, std::size_t, Vec3f, Vec3f, f32);
__global__ void density_change_kernel(
    const Vec3f*, const Vec3f*, const f32*, const u32*, const u32*,
    f32*, f32*, const f32*, f32, std::size_t,
    Vec3f, Vec3i, f32, f32, f32, f32);
__global__ void apply_kappa_kernel(
    const Vec3f*, Vec3f*, const f32*, const f32*, const u32*, const u32*,
    std::size_t, Vec3f, Vec3i, f32, f32, f32, f32);
}


namespace {

CellGrid make_grid(const DFSPHSolver::Config& cfg, cudaStream_t stream) {
    Vec3i dims;
    dims.x = static_cast<i32>(std::ceil((cfg.domain_max.x - cfg.domain_min.x) / cfg.smoothing_length));
    dims.y = static_cast<i32>(std::ceil((cfg.domain_max.y - cfg.domain_min.y) / cfg.smoothing_length));
    dims.z = static_cast<i32>(std::ceil((cfg.domain_max.z - cfg.domain_min.z) / cfg.smoothing_length));
    return CellGrid(cfg.domain_min, dims, cfg.smoothing_length, stream);
}

} // namespace

DFSPHSolver::DFSPHSolver(ParticleStore& store, Config cfg, cudaStream_t stream)
    : store_(store), cfg_(cfg), stream_(stream),
      grid_(make_grid(cfg, stream)) {

    const f32 spacing = 2.0f * cfg_.particle_radius;
    const f32 v0      = spacing * spacing * spacing;
    mass_             = cfg_.rest_density * v0;

    density_     = store_.register_attribute<f32>("dfsph_density",     AttribType::F32);
    alpha_       = store_.register_attribute<f32>("dfsph_alpha",       AttribType::F32);
    kappa_       = store_.register_attribute<f32>("dfsph_kappa",       AttribType::F32);
    kappa_v_     = store_.register_attribute<f32>("dfsph_kappa_v",     AttribType::F32);
    density_adv_ = store_.register_attribute<f32>("dfsph_density_adv", AttribType::F32);

    grid_.reserve(store_.capacity());

    reduce_in_  = DeviceBuffer<f32>(store_.capacity(), stream_);
    reduce_out_ = DeviceBuffer<f32>(1, stream_);
}

void DFSPHSolver::step(f32 dt) {
    grid_.build(store_.positions(), store_.count());
    compute_density();
    compute_alpha();
    apply_external_forces(dt);
    density_solve(dt);
    advect(dt);

    // Phase 2 placeholder for max_velocity_ (proper GPU reduction is T10).
    // A constant ~1 m/s is a reasonable CFL hint for slow-falling water and
    // avoids the self-amplifying bug from a naive accumulator.
    max_velocity_ = 1.0f;
}

f32 DFSPHSolver::density_at(std::size_t i) const {
    f32 v = 0.0f;
    WATER_CUDA_CHECK(cudaMemcpy(&v,
        const_cast<ParticleStore&>(store_).attribute_data(density_) + i,
        sizeof(f32), cudaMemcpyDeviceToHost));
    return v;
}

f32 DFSPHSolver::alpha_at(std::size_t i) const {
    f32 v = 0.0f;
    WATER_CUDA_CHECK(cudaMemcpy(&v,
        const_cast<ParticleStore&>(store_).attribute_data(alpha_) + i,
        sizeof(f32), cudaMemcpyDeviceToHost));
    return v;
}

void DFSPHSolver::compute_density() {
    const std::size_t n = store_.count();
    if (n == 0) return;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::density_kernel<<<grid, block, 0, stream_>>>(
        store_.positions(),
        grid_.sorted_indices(),
        grid_.cell_start(),
        store_.attribute_data(density_),
        n,
        grid_.origin(),
        grid_.cells_per_axis(),
        grid_.cell_length(),
        mass_,
        cfg_.smoothing_length);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::compute_alpha() {
    const std::size_t n = store_.count();
    if (n == 0) return;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::alpha_kernel<<<grid, block, 0, stream_>>>(
        store_.positions(),
        store_.attribute_data(density_),
        grid_.sorted_indices(),
        grid_.cell_start(),
        store_.attribute_data(alpha_),
        n,
        grid_.origin(),
        grid_.cells_per_axis(),
        grid_.cell_length(),
        mass_,
        cfg_.smoothing_length);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::apply_external_forces(f32 dt) {
    const std::size_t n = store_.count();
    if (n == 0) return;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::apply_gravity_kernel<<<grid, block, 0, stream_>>>(
        store_.velocities(), n, cfg_.gravity, dt);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::advect(f32 dt) {
    const std::size_t n = store_.count();
    if (n == 0) return;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::advect_kernel<<<grid, block, 0, stream_>>>(
        store_.positions(), store_.velocities(), n,
        cfg_.domain_min, cfg_.domain_max, dt);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::density_solve(f32 dt) {
    const std::size_t n = store_.count();
    if (n == 0) return;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    // Phase 2 (interim): fixed iteration count (5). The CUB-based convergence
    // check is the second half of T7/T8 — added when needed for cinematic
    // quality. Five iterations is sufficient for visible incompressibility.
    // 20 fixed iterations is enough to support a settled fluid stack
    // visually. The CUB-based convergence check (T8) replaces this with
    // an early-exit when mean density error drops below eta_density.
    const u32 iters = std::min<u32>(20u, cfg_.max_density_iters);
    for (u32 iter = 0; iter < iters; ++iter) {
        detail::density_change_kernel<<<grid, block, 0, stream_>>>(
            store_.positions(), store_.velocities(),
            store_.attribute_data(density_),
            grid_.sorted_indices(), grid_.cell_start(),
            store_.attribute_data(density_adv_), store_.attribute_data(kappa_),
            store_.attribute_data(alpha_),
            cfg_.rest_density,
            n, grid_.origin(), grid_.cells_per_axis(), grid_.cell_length(),
            mass_, cfg_.smoothing_length, dt);
        WATER_CUDA_CHECK_LAST();

        detail::apply_kappa_kernel<<<grid, block, 0, stream_>>>(
            store_.positions(), store_.velocities(),
            store_.attribute_data(density_), store_.attribute_data(kappa_),
            grid_.sorted_indices(), grid_.cell_start(),
            n, grid_.origin(), grid_.cells_per_axis(), grid_.cell_length(),
            mass_, cfg_.smoothing_length, dt);
        WATER_CUDA_CHECK_LAST();
    }
}

// Stub host methods — populated in subsequent tasks.
void DFSPHSolver::surface_tension(f32 /*dt*/) {}
void DFSPHSolver::divergence_solve(f32 /*dt*/) {}
f32  DFSPHSolver::mean_density_error_()    { return 0.0f; }
f32  DFSPHSolver::mean_divergence_error_() { return 0.0f; }

} // namespace water::solvers

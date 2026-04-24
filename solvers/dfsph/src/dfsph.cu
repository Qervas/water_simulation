#include "water/solvers/dfsph.h"
#include "water/core/cuda_check.h"
#include <cmath>

namespace water::solvers {

namespace detail {
__global__ void density_kernel(
    const Vec3f*, const u32*, const u32*, f32*,
    std::size_t, Vec3f, Vec3i, f32, f32, f32);
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

void DFSPHSolver::step(f32 /*dt*/) {
    grid_.build(store_.positions(), store_.count());
    compute_density();
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

// Stub host methods — populated in subsequent tasks.
void DFSPHSolver::compute_alpha() {}
void DFSPHSolver::apply_external_forces(f32 /*dt*/) {}
void DFSPHSolver::surface_tension(f32 /*dt*/) {}
void DFSPHSolver::density_solve(f32 /*dt*/) {}
void DFSPHSolver::advect(f32 /*dt*/) {}
void DFSPHSolver::divergence_solve(f32 /*dt*/) {}
f32  DFSPHSolver::mean_density_error_()    { return 0.0f; }
f32  DFSPHSolver::mean_divergence_error_() { return 0.0f; }

} // namespace water::solvers

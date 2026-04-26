#include "water/solvers/dfsph.h"
#include "water/core/cuda_check.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>

namespace water::solvers {

// Forward decls of kernels in dfsph_kernels.cu
namespace detail {
__global__ void compute_boundary_mass_kernel(
    const Vec3f*, const u32*, const u32*, f32*, std::size_t,
    Vec3f, Vec3i, f32, f32, f32);

__global__ void density_kernel(
    const Vec3f*, const u32*, const u32*, f32*, std::size_t,
    Vec3f, Vec3i, f32, f32, f32,
    const Vec3f*, const u32*, const u32*, const f32*,
    Vec3f, Vec3i, f32, std::size_t);

__global__ void alpha_kernel(
    const Vec3f*, const f32*, const u32*, const u32*, f32*, std::size_t,
    Vec3f, Vec3i, f32, f32, f32,
    const Vec3f*, const u32*, const u32*, const f32*,
    Vec3f, Vec3i, f32, std::size_t);

__global__ void density_change_kernel(
    const Vec3f*, const Vec3f*, const f32*, const u32*, const u32*,
    f32*, f32*, const f32*, f32, std::size_t,
    Vec3f, Vec3i, f32, f32, f32, f32,
    const Vec3f*, const u32*, const u32*, const f32*,
    Vec3f, Vec3i, f32, std::size_t);

__global__ void apply_kappa_kernel(
    const Vec3f*, Vec3f*, const f32*, const f32*, const u32*, const u32*,
    std::size_t,
    Vec3f, Vec3i, f32, f32, f32, f32, f32,
    const Vec3f*, const u32*, const u32*, const f32*,
    Vec3f, Vec3i, f32, std::size_t);

__global__ void xsph_viscosity_kernel(
    const Vec3f*, Vec3f*, const f32*, const u32*, const u32*, std::size_t,
    Vec3f, Vec3i, f32, f32, f32, f32);

__global__ void apply_gravity_kernel(Vec3f*, std::size_t, Vec3f, f32, f32);
__global__ void advect_kernel(Vec3f*, Vec3f*, std::size_t, f32, f32, Vec3f, Vec3f);
__global__ void density_error_kernel(const f32*, f32, f32*, std::size_t);
__global__ void velocity_magnitude_kernel(const Vec3f*, f32*, std::size_t);
}

// ---------- helpers ----------

namespace {

CellGrid make_grid(const DFSPHSolver::Config& cfg, cudaStream_t stream) {
    Vec3i dims;
    dims.x = static_cast<i32>(std::ceil((cfg.domain_max.x - cfg.domain_min.x) / cfg.smoothing_length)) + 2;
    dims.y = static_cast<i32>(std::ceil((cfg.domain_max.y - cfg.domain_min.y) / cfg.smoothing_length)) + 2;
    dims.z = static_cast<i32>(std::ceil((cfg.domain_max.z - cfg.domain_min.z) / cfg.smoothing_length)) + 2;
    // Origin shifted by one cell so particles slightly outside domain still hit valid cells.
    Vec3f origin{cfg.domain_min.x - cfg.smoothing_length,
                 cfg.domain_min.y - cfg.smoothing_length,
                 cfg.domain_min.z - cfg.smoothing_length};
    return CellGrid(origin, dims, cfg.smoothing_length, stream);
}

} // namespace

// ---------- ctor / mass calibration ----------

DFSPHSolver::DFSPHSolver(ParticleStore& fluid, ParticleStore* boundary,
                          Config cfg, cudaStream_t stream)
    : fluid_(fluid), boundary_(boundary), cfg_(cfg), stream_(stream),
      fluid_grid_(make_grid(cfg, stream)) {

    const f32 spacing = 2.0f * cfg_.particle_radius;
    const f32 v0      = spacing * spacing * spacing;
    mass_             = cfg_.rest_density * v0;

    density_     = fluid_.register_attribute<f32>("dfsph_density",     AttribType::F32);
    alpha_       = fluid_.register_attribute<f32>("dfsph_alpha",       AttribType::F32);
    kappa_       = fluid_.register_attribute<f32>("dfsph_kappa",       AttribType::F32);
    density_adv_ = fluid_.register_attribute<f32>("dfsph_density_adv", AttribType::F32);

    fluid_grid_.reserve(fluid_.capacity());

    reduce_in_  = DeviceBuffer<f32>(fluid_.capacity(), stream_);
    reduce_out_ = DeviceBuffer<f32>(1, stream_);

    if (boundary_) {
        boundary_mass_ = boundary_->register_attribute<f32>(
            "dfsph_boundary_mass", AttribType::F32);
        boundary_grid_ = std::make_unique<CellGrid>(make_grid(cfg, stream));
        boundary_grid_->reserve(boundary_->capacity());

        // Build the boundary cell grid once (boundary particles never move).
        boundary_grid_->build(boundary_->positions(), boundary_->count());
        compute_boundary_mass();
    }
}

void DFSPHSolver::compute_boundary_mass() {
    if (!boundary_ || boundary_->count() == 0) return;
    const std::size_t n = boundary_->count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::compute_boundary_mass_kernel<<<grid, block, 0, stream_>>>(
        boundary_->positions(),
        boundary_grid_->sorted_indices(), boundary_grid_->cell_start(),
        boundary_->attribute_data(boundary_mass_),
        n,
        boundary_grid_->origin(), boundary_grid_->cells_per_axis(),
        boundary_grid_->cell_length(),
        cfg_.rest_density, cfg_.smoothing_length);
    WATER_CUDA_CHECK_LAST();
}

// ---------- per-step pipeline ----------

void DFSPHSolver::step(f32 dt) {
    const std::size_t n = fluid_.count();
    if (n == 0) return;

    fluid_grid_.build(fluid_.positions(), n);
    compute_density();
    compute_alpha();
    apply_external_forces(dt);
    apply_xsph_viscosity(dt);
    density_solve(dt);
    advect(dt);

    max_velocity_ = reduce_max_velocity_();
}

// ---------- per-substep stages ----------

void DFSPHSolver::compute_density() {
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);

    const Vec3f*  pb = boundary_ ? boundary_->positions() : nullptr;
    const u32*    sb = boundary_grid_ ? boundary_grid_->sorted_indices() : nullptr;
    const u32*    cb = boundary_grid_ ? boundary_grid_->cell_start()     : nullptr;
    const f32*    mb = boundary_ ? boundary_->attribute_data(boundary_mass_) : nullptr;
    const Vec3f   ob = boundary_grid_ ? boundary_grid_->origin()         : Vec3f{0,0,0};
    const Vec3i   db = boundary_grid_ ? boundary_grid_->cells_per_axis() : Vec3i{0,0,0};
    const f32     lb = boundary_grid_ ? boundary_grid_->cell_length()    : 0.0f;
    const std::size_t nb = boundary_ ? boundary_->count() : 0;

    detail::density_kernel<<<grid, block, 0, stream_>>>(
        fluid_.positions(),
        fluid_grid_.sorted_indices(), fluid_grid_.cell_start(),
        fluid_.attribute_data(density_),
        n,
        fluid_grid_.origin(), fluid_grid_.cells_per_axis(), fluid_grid_.cell_length(),
        mass_, cfg_.smoothing_length,
        pb, sb, cb, mb, ob, db, lb, nb);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::compute_alpha() {
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);

    const Vec3f*  pb = boundary_ ? boundary_->positions() : nullptr;
    const u32*    sb = boundary_grid_ ? boundary_grid_->sorted_indices() : nullptr;
    const u32*    cb = boundary_grid_ ? boundary_grid_->cell_start()     : nullptr;
    const f32*    mb = boundary_ ? boundary_->attribute_data(boundary_mass_) : nullptr;
    const Vec3f   ob = boundary_grid_ ? boundary_grid_->origin()         : Vec3f{0,0,0};
    const Vec3i   db = boundary_grid_ ? boundary_grid_->cells_per_axis() : Vec3i{0,0,0};
    const f32     lb = boundary_grid_ ? boundary_grid_->cell_length()    : 0.0f;
    const std::size_t nb = boundary_ ? boundary_->count() : 0;

    detail::alpha_kernel<<<grid, block, 0, stream_>>>(
        fluid_.positions(),
        fluid_.attribute_data(density_),
        fluid_grid_.sorted_indices(), fluid_grid_.cell_start(),
        fluid_.attribute_data(alpha_),
        n,
        fluid_grid_.origin(), fluid_grid_.cells_per_axis(), fluid_grid_.cell_length(),
        mass_, cfg_.smoothing_length,
        pb, sb, cb, mb, ob, db, lb, nb);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::apply_external_forces(f32 dt) {
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::apply_gravity_kernel<<<grid, block, 0, stream_>>>(
        fluid_.velocities(), n, cfg_.gravity, dt, cfg_.damping);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::apply_xsph_viscosity(f32 dt) {
    if (cfg_.viscosity <= 0.0f) return;
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    // XSPH is "epsilon * (smooth velocity toward neighborhood mean)" — dt-independent
    // in the common formulation; we ignore dt here.
    (void)dt;
    detail::xsph_viscosity_kernel<<<grid, block, 0, stream_>>>(
        fluid_.positions(), fluid_.velocities(),
        fluid_.attribute_data(density_),
        fluid_grid_.sorted_indices(), fluid_grid_.cell_start(),
        n,
        fluid_grid_.origin(), fluid_grid_.cells_per_axis(), fluid_grid_.cell_length(),
        mass_, cfg_.smoothing_length, cfg_.viscosity);
    WATER_CUDA_CHECK_LAST();
}

void DFSPHSolver::density_solve(f32 dt) {
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);

    const Vec3f*  pb = boundary_ ? boundary_->positions() : nullptr;
    const u32*    sb = boundary_grid_ ? boundary_grid_->sorted_indices() : nullptr;
    const u32*    cb = boundary_grid_ ? boundary_grid_->cell_start()     : nullptr;
    const f32*    mb = boundary_ ? boundary_->attribute_data(boundary_mass_) : nullptr;
    const Vec3f   ob = boundary_grid_ ? boundary_grid_->origin()         : Vec3f{0,0,0};
    const Vec3i   db = boundary_grid_ ? boundary_grid_->cells_per_axis() : Vec3i{0,0,0};
    const f32     lb = boundary_grid_ ? boundary_grid_->cell_length()    : 0.0f;
    const std::size_t nb = boundary_ ? boundary_->count() : 0;

    last_density_iters_ = 0;
    for (u32 iter = 0; iter < cfg_.max_density_iters; ++iter) {
        detail::density_change_kernel<<<grid, block, 0, stream_>>>(
            fluid_.positions(), fluid_.velocities(),
            fluid_.attribute_data(density_),
            fluid_grid_.sorted_indices(), fluid_grid_.cell_start(),
            fluid_.attribute_data(density_adv_), fluid_.attribute_data(kappa_),
            fluid_.attribute_data(alpha_),
            cfg_.rest_density,
            n,
            fluid_grid_.origin(), fluid_grid_.cells_per_axis(), fluid_grid_.cell_length(),
            mass_, cfg_.smoothing_length, dt,
            pb, sb, cb, mb, ob, db, lb, nb);
        WATER_CUDA_CHECK_LAST();

        detail::apply_kappa_kernel<<<grid, block, 0, stream_>>>(
            fluid_.positions(), fluid_.velocities(),
            fluid_.attribute_data(density_), fluid_.attribute_data(kappa_),
            fluid_grid_.sorted_indices(), fluid_grid_.cell_start(),
            n,
            fluid_grid_.origin(), fluid_grid_.cells_per_axis(), fluid_grid_.cell_length(),
            mass_, cfg_.smoothing_length, dt, cfg_.rest_density,
            pb, sb, cb, mb, ob, db, lb, nb);
        WATER_CUDA_CHECK_LAST();

        last_density_iters_ = iter + 1;
        if (iter >= 1) {
            const f32 err = reduce_density_error_();
            if (err < cfg_.eta_density * cfg_.rest_density) break;
        }
    }
}

void DFSPHSolver::advect(f32 dt) {
    const std::size_t n = fluid_.count();
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    // Hard velocity clamp to keep DFSPH stable at high impact velocities.
    // 5 m/s is well within "fast splash" territory while still bounded.
    constexpr f32 MAX_VELOCITY_CLAMP = 5.0f;
    detail::advect_kernel<<<grid, block, 0, stream_>>>(
        fluid_.positions(), fluid_.velocities(), n, dt, MAX_VELOCITY_CLAMP,
        cfg_.domain_min, cfg_.domain_max);
    WATER_CUDA_CHECK_LAST();
}

// ---------- CUB reductions ----------

f32 DFSPHSolver::reduce_density_error_() {
    const std::size_t n = fluid_.count();
    if (n == 0) return 0.0f;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::density_error_kernel<<<grid, block, 0, stream_>>>(
        fluid_.attribute_data(density_adv_), cfg_.rest_density,
        reduce_in_.data(), n);
    WATER_CUDA_CHECK_LAST();

    std::size_t bytes = 0;
    cub::DeviceReduce::Sum(nullptr, bytes,
        reduce_in_.data(), reduce_out_.data(), static_cast<int>(n), stream_);
    if (bytes > reduce_temp_.size()) {
        reduce_temp_ = DeviceBuffer<unsigned char>(bytes, stream_);
    }
    cub::DeviceReduce::Sum(reduce_temp_.data(), bytes,
        reduce_in_.data(), reduce_out_.data(), static_cast<int>(n), stream_);

    f32 sum_h = 0.f;
    WATER_CUDA_CHECK(cudaMemcpyAsync(&sum_h, reduce_out_.data(), sizeof(f32),
                                      cudaMemcpyDeviceToHost, stream_));
    WATER_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return sum_h / static_cast<f32>(n);
}

f32 DFSPHSolver::reduce_max_velocity_() {
    const std::size_t n = fluid_.count();
    if (n == 0) return 0.0f;
    constexpr int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    detail::velocity_magnitude_kernel<<<grid, block, 0, stream_>>>(
        fluid_.velocities(), reduce_in_.data(), n);
    WATER_CUDA_CHECK_LAST();

    std::size_t bytes = 0;
    cub::DeviceReduce::Max(nullptr, bytes,
        reduce_in_.data(), reduce_out_.data(), static_cast<int>(n), stream_);
    if (bytes > reduce_temp_.size()) {
        reduce_temp_ = DeviceBuffer<unsigned char>(bytes, stream_);
    }
    cub::DeviceReduce::Max(reduce_temp_.data(), bytes,
        reduce_in_.data(), reduce_out_.data(), static_cast<int>(n), stream_);

    f32 mv = 0.f;
    WATER_CUDA_CHECK(cudaMemcpyAsync(&mv, reduce_out_.data(), sizeof(f32),
                                      cudaMemcpyDeviceToHost, stream_));
    WATER_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return mv;
}

// ---------- test/diagnostic accessors ----------

f32 DFSPHSolver::density_at(std::size_t i) const {
    f32 v = 0.0f;
    WATER_CUDA_CHECK(cudaMemcpy(&v,
        const_cast<ParticleStore&>(fluid_).attribute_data(density_) + i,
        sizeof(f32), cudaMemcpyDeviceToHost));
    return v;
}

f32 DFSPHSolver::alpha_at(std::size_t i) const {
    f32 v = 0.0f;
    WATER_CUDA_CHECK(cudaMemcpy(&v,
        const_cast<ParticleStore&>(fluid_).attribute_data(alpha_) + i,
        sizeof(f32), cudaMemcpyDeviceToHost));
    return v;
}

f32 DFSPHSolver::boundary_mass_at(std::size_t i) const {
    if (!boundary_) return 0.0f;
    f32 v = 0.0f;
    WATER_CUDA_CHECK(cudaMemcpy(&v,
        const_cast<ParticleStore&>(*boundary_).attribute_data(boundary_mass_) + i,
        sizeof(f32), cudaMemcpyDeviceToHost));
    return v;
}

} // namespace water::solvers

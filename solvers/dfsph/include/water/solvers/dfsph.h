#pragma once

#include "water/core/cell_grid.h"
#include "water/core/particle_store.h"
#include "water/core/types.h"
#include <memory>

namespace water::solvers {

// Divergence-Free SPH (Bender & Koschier 2015) with Akinci 2012 boundary
// particles, XSPH viscosity, and CUB-based density-solver convergence.
//
// Solver registers per-particle attributes on the fluid ParticleStore at
// construction. If a boundary ParticleStore is supplied, it is also augmented
// with a calibrated mass attribute and gets its own CellGrid; both fluid and
// boundary participate in density/pressure neighbor sums.
class DFSPHSolver {
public:
    struct Config {
        f32   rest_density        = 1000.0f;
        f32   particle_radius     = 0.0015f;
        f32   smoothing_length    = 0.003f;
        f32   viscosity           = 1e-3f;       // XSPH velocity smoothing coefficient
        f32   surface_tension     = 0.0f;
        Vec3f gravity             {0.0f, -9.81f, 0.0f};
        u32   max_density_iters   = 100;
        u32   max_divergence_iters= 100;
        f32   eta_density         = 1e-3f;
        f32   eta_divergence      = 1e-3f;
        Vec3f domain_min          {0.0f, 0.0f, 0.0f};
        Vec3f domain_max          {1.0f, 1.0f, 1.0f};
    };

    // boundary may be nullptr (then solver runs without proper boundary forces).
    DFSPHSolver(ParticleStore& fluid, ParticleStore* boundary,
                Config cfg, cudaStream_t stream = 0);

    void step(f32 dt);

    f32 max_velocity() const noexcept { return max_velocity_; }

    const CellGrid& fluid_grid()    const noexcept { return fluid_grid_; }
    const CellGrid* boundary_grid() const noexcept { return boundary_grid_.get(); }
    f32             particle_mass() const noexcept { return mass_; }

    // Test/diagnostic helpers — synchronous device→host reads.
    f32 density_at(std::size_t i) const;
    f32 alpha_at(std::size_t i) const;
    f32 boundary_mass_at(std::size_t i) const;
    u32 last_density_iters() const noexcept { return last_density_iters_; }

private:
    void compute_boundary_mass();
    void compute_density();
    void compute_alpha();
    void apply_external_forces(f32 dt);
    void apply_xsph_viscosity(f32 dt);
    void density_solve(f32 dt);
    void advect(f32 dt);
    f32  reduce_density_error_();
    f32  reduce_max_velocity_();

    ParticleStore& fluid_;
    ParticleStore* boundary_;          // may be null
    Config         cfg_;
    cudaStream_t   stream_;

    CellGrid                     fluid_grid_;
    std::unique_ptr<CellGrid>    boundary_grid_;     // built once, never updated
    f32                          mass_;              // fluid mass per particle
    f32                          max_velocity_ = 0.0f;
    u32                          last_density_iters_ = 0;

    AttribHandle<f32>   density_;
    AttribHandle<f32>   alpha_;
    AttribHandle<f32>   kappa_;
    AttribHandle<f32>   density_adv_;
    AttribHandle<f32>   boundary_mass_;   // valid only if boundary != null

    DeviceBuffer<f32>           reduce_in_;
    DeviceBuffer<f32>           reduce_out_;
    DeviceBuffer<unsigned char> reduce_temp_;
};

} // namespace water::solvers

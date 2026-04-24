#pragma once

#include "water/core/cell_grid.h"
#include "water/core/particle_store.h"
#include "water/core/types.h"

namespace water::solvers {

// Divergence-Free SPH (Bender & Koschier 2015) with Akinci 2013 surface
// tension. Solver registers per-particle attributes on the supplied
// ParticleStore at construction.
class DFSPHSolver {
public:
    struct Config {
        f32   rest_density        = 1000.0f;
        f32   particle_radius     = 0.0015f;
        f32   smoothing_length    = 0.003f;
        f32   viscosity           = 1e-3f;
        f32   surface_tension     = 0.0728f;
        Vec3f gravity             {0.0f, -9.81f, 0.0f};
        u32   max_density_iters   = 100;
        u32   max_divergence_iters= 100;
        f32   eta_density         = 1e-3f;
        f32   eta_divergence      = 1e-3f;
        Vec3f domain_min          {0.0f, 0.0f, 0.0f};
        Vec3f domain_max          {1.0f, 1.0f, 1.0f};
    };

    DFSPHSolver(ParticleStore& store, Config cfg, cudaStream_t stream = 0);

    // Advance the simulation by `dt` seconds.
    void step(f32 dt);

    // Maximum particle speed after the most recent step (CFL hint).
    f32 max_velocity() const noexcept { return max_velocity_; }

    const CellGrid& cell_grid()    const noexcept { return grid_; }
    f32             particle_mass() const noexcept { return mass_; }

    // Test helpers — synchronous device→host reads.
    f32 density_at(std::size_t i) const;
    f32 alpha_at(std::size_t i) const;

private:
    void compute_density();
    void compute_alpha();
    void apply_external_forces(f32 dt);
    void surface_tension(f32 dt);
    void density_solve(f32 dt);
    void advect(f32 dt);
    void divergence_solve(f32 dt);
    f32  mean_density_error_();
    f32  mean_divergence_error_();

    ParticleStore& store_;
    Config         cfg_;
    cudaStream_t   stream_;
    CellGrid       grid_;
    f32            mass_;
    f32            max_velocity_ = 0.0f;

    AttribHandle<f32>   density_;
    AttribHandle<f32>   alpha_;
    AttribHandle<f32>   kappa_;
    AttribHandle<f32>   kappa_v_;
    AttribHandle<f32>   density_adv_;

    DeviceBuffer<f32>           reduce_in_;
    DeviceBuffer<f32>           reduce_out_;
    DeviceBuffer<unsigned char> reduce_temp_;
};

} // namespace water::solvers

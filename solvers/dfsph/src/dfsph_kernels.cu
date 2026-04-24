#include "water/core/sph_kernels.cuh"
#include "water/core/cuda_check.h"
#include "water/solvers/dfsph.h"
#include <cuda_runtime.h>

namespace water::solvers::detail {

__global__ void density_kernel(
        const Vec3f* positions,
        const u32* sorted_indices,
        const u32* cell_start,
        f32* density_out,
        std::size_t n_particles,
        Vec3f origin, Vec3i dims, f32 cell_length, f32 mass, f32 smoothing_length) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    const Vec3f p_i = positions[i];
    const f32 inv_cell = 1.0f / cell_length;
    const int cx = static_cast<int>(floorf((p_i.x - origin.x) * inv_cell));
    const int cy = static_cast<int>(floorf((p_i.y - origin.y) * inv_cell));
    const int cz = static_cast<int>(floorf((p_i.z - origin.z) * inv_cell));

    f32 rho = 0.0f;
    const u32 sentinel = static_cast<u32>(dims.x) * static_cast<u32>(dims.y)
                       * static_cast<u32>(dims.z);

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= dims.x || ny < 0 || ny >= dims.y ||
            nz < 0 || nz >= dims.z) continue;
        const u32 c = (static_cast<u32>(nx) * static_cast<u32>(dims.y)
                     + static_cast<u32>(ny)) * static_cast<u32>(dims.z)
                     + static_cast<u32>(nz);
        if (c >= sentinel) continue;
        const u32 begin = cell_start[c];
        const u32 end   = cell_start[c + 1];
        for (u32 s = begin; s < end; ++s) {
            const u32 j = sorted_indices[s];
            const Vec3f p_j = positions[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const f32 dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
            rho += mass * sph::cubic_spline_W(dist, smoothing_length);
        }
    }
    density_out[i] = rho;
}

__global__ void alpha_kernel(
        const Vec3f* positions,
        const f32* density,
        const u32* sorted_indices,
        const u32* cell_start,
        f32* alpha_out,
        std::size_t n_particles,
        Vec3f origin, Vec3i dims, f32 cell_length, f32 mass, f32 smoothing_length) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    const Vec3f p_i = positions[i];
    const f32 inv_cell = 1.0f / cell_length;
    const int cx = static_cast<int>(floorf((p_i.x - origin.x) * inv_cell));
    const int cy = static_cast<int>(floorf((p_i.y - origin.y) * inv_cell));
    const int cz = static_cast<int>(floorf((p_i.z - origin.z) * inv_cell));

    Vec3f sum_grad{0.f, 0.f, 0.f};
    f32   sum_grad_sq = 0.0f;
    const u32 sentinel = static_cast<u32>(dims.x) * static_cast<u32>(dims.y)
                       * static_cast<u32>(dims.z);

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= dims.x || ny < 0 || ny >= dims.y ||
            nz < 0 || nz >= dims.z) continue;
        const u32 c = (static_cast<u32>(nx) * static_cast<u32>(dims.y)
                     + static_cast<u32>(ny)) * static_cast<u32>(dims.z)
                     + static_cast<u32>(nz);
        if (c >= sentinel) continue;
        const u32 begin = cell_start[c];
        const u32 end   = cell_start[c + 1];
        for (u32 s = begin; s < end; ++s) {
            const u32 j = sorted_indices[s];
            if (j == i) continue;
            const Vec3f p_j = positions[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const Vec3f mgW{mass * gW.x, mass * gW.y, mass * gW.z};
            sum_grad.x += mgW.x; sum_grad.y += mgW.y; sum_grad.z += mgW.z;
            sum_grad_sq += mgW.x * mgW.x + mgW.y * mgW.y + mgW.z * mgW.z;
        }
    }

    const f32 denom = sum_grad.x * sum_grad.x + sum_grad.y * sum_grad.y
                    + sum_grad.z * sum_grad.z + sum_grad_sq;
    // Bender-Koschier 2015 eq. 11. Guard against denom=0 for isolated particles.
    alpha_out[i] = (denom > 1e-9f) ? (density[i] / denom) : 0.0f;
}

__global__ void density_change_kernel(
        const Vec3f* positions, const Vec3f* velocities,
        const f32* density,
        const u32* sorted_indices, const u32* cell_start,
        f32* density_adv_out, f32* kappa_out,
        const f32* alpha,
        f32 rest_density,
        std::size_t n_particles,
        Vec3f origin, Vec3i dims, f32 cell_length,
        f32 mass, f32 smoothing_length, f32 dt) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    const Vec3f p_i = positions[i];
    const Vec3f v_i = velocities[i];
    const f32 inv_cell = 1.0f / cell_length;
    const int cx = static_cast<int>(floorf((p_i.x - origin.x) * inv_cell));
    const int cy = static_cast<int>(floorf((p_i.y - origin.y) * inv_cell));
    const int cz = static_cast<int>(floorf((p_i.z - origin.z) * inv_cell));

    f32 rho_dot = 0.0f;
    const u32 sentinel = static_cast<u32>(dims.x) * static_cast<u32>(dims.y)
                       * static_cast<u32>(dims.z);

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= dims.x || ny < 0 || ny >= dims.y ||
            nz < 0 || nz >= dims.z) continue;
        const u32 c = (static_cast<u32>(nx) * static_cast<u32>(dims.y)
                     + static_cast<u32>(ny)) * static_cast<u32>(dims.z)
                     + static_cast<u32>(nz);
        if (c >= sentinel) continue;
        const u32 begin = cell_start[c];
        const u32 end   = cell_start[c + 1];
        for (u32 s = begin; s < end; ++s) {
            const u32 j = sorted_indices[s];
            if (j == i) continue;
            const Vec3f p_j = positions[j];
            const Vec3f v_j = velocities[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const Vec3f dv{v_i.x - v_j.x, v_i.y - v_j.y, v_i.z - v_j.z};
            rho_dot += mass * (dv.x * gW.x + dv.y * gW.y + dv.z * gW.z);
        }
    }

    const f32 rho_pred = density[i] + dt * rho_dot;
    density_adv_out[i] = rho_pred;
    // Bender-Koschier 2015: kappa allows BOTH positive (compression) and
    // negative (rarefaction) corrections. Clamping to >=0 prevents the
    // solver from re-densifying an under-packed fluid, which manifests
    // visually as particles never stacking.
    const f32 s = (rho_pred - rest_density) / dt;
    kappa_out[i] = s * alpha[i] / dt;
}

__global__ void apply_kappa_kernel(
        const Vec3f* positions,
        Vec3f* velocities,
        const f32* density,
        const f32* kappa,
        const u32* sorted_indices, const u32* cell_start,
        std::size_t n_particles,
        Vec3f origin, Vec3i dims, f32 cell_length,
        f32 mass, f32 smoothing_length, f32 dt) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    const Vec3f p_i = positions[i];
    const f32 rho_i = density[i];
    const f32 k_i   = kappa[i];
    const f32 inv_cell = 1.0f / cell_length;
    const int cx = static_cast<int>(floorf((p_i.x - origin.x) * inv_cell));
    const int cy = static_cast<int>(floorf((p_i.y - origin.y) * inv_cell));
    const int cz = static_cast<int>(floorf((p_i.z - origin.z) * inv_cell));

    Vec3f delta_v{0.f, 0.f, 0.f};
    const u32 sentinel = static_cast<u32>(dims.x) * static_cast<u32>(dims.y)
                       * static_cast<u32>(dims.z);

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || nx >= dims.x || ny < 0 || ny >= dims.y ||
            nz < 0 || nz >= dims.z) continue;
        const u32 c = (static_cast<u32>(nx) * static_cast<u32>(dims.y)
                     + static_cast<u32>(ny)) * static_cast<u32>(dims.z)
                     + static_cast<u32>(nz);
        if (c >= sentinel) continue;
        const u32 begin = cell_start[c];
        const u32 end   = cell_start[c + 1];
        for (u32 s = begin; s < end; ++s) {
            const u32 j = sorted_indices[s];
            if (j == i) continue;
            const Vec3f p_j = positions[j];
            const f32 rho_j = density[j];
            const f32 k_j   = kappa[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const f32 ki = k_i / fmaxf(rho_i * rho_i, 1e-9f);
            const f32 kj = k_j / fmaxf(rho_j * rho_j, 1e-9f);
            const f32 c2 = mass * (ki + kj);
            delta_v.x -= dt * c2 * gW.x;
            delta_v.y -= dt * c2 * gW.y;
            delta_v.z -= dt * c2 * gW.z;
        }
    }

    velocities[i].x += delta_v.x;
    velocities[i].y += delta_v.y;
    velocities[i].z += delta_v.z;
}

__global__ void apply_gravity_kernel(Vec3f* velocity, std::size_t n,
                                      Vec3f g, f32 dt) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    velocity[i].x += g.x * dt;
    velocity[i].y += g.y * dt;
    velocity[i].z += g.z * dt;
}

__global__ void advect_kernel(Vec3f* position, Vec3f* velocity, std::size_t n,
                               Vec3f domain_min, Vec3f domain_max, f32 dt) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3f p{position[i].x + velocity[i].x * dt,
            position[i].y + velocity[i].y * dt,
            position[i].z + velocity[i].z * dt};
    Vec3f v = velocity[i];
    const f32 eps      = 1e-3f;
    const f32 restitution = 0.4f;  // velocity reflection coefficient on bounce
    // Reflect velocity off walls (with damping) so stacked fluid can be
    // supported by floor reaction. Phase 4 replaces with proper SDF boundary.
    if (p.x <= domain_min.x + eps) { p.x = domain_min.x + eps; if (v.x < 0) v.x = -restitution * v.x; }
    if (p.y <= domain_min.y + eps) { p.y = domain_min.y + eps; if (v.y < 0) v.y = -restitution * v.y; }
    if (p.z <= domain_min.z + eps) { p.z = domain_min.z + eps; if (v.z < 0) v.z = -restitution * v.z; }
    if (p.x >= domain_max.x - eps) { p.x = domain_max.x - eps; if (v.x > 0) v.x = -restitution * v.x; }
    if (p.y >= domain_max.y - eps) { p.y = domain_max.y - eps; if (v.y > 0) v.y = -restitution * v.y; }
    if (p.z >= domain_max.z - eps) { p.z = domain_max.z - eps; if (v.z > 0) v.z = -restitution * v.z; }
    position[i] = p;
    velocity[i] = v;
}

} // namespace water::solvers::detail

#include "water/core/sph_kernels.cuh"
#include "water/core/cuda_check.h"
#include "water/solvers/dfsph.h"
#include <cuda_runtime.h>

namespace water::solvers::detail {

// ---------- helpers ----------

// 27-cell stencil iteration, called from each kernel below.
// Body is the variadic last "argument" so commas inside struct inits are fine.
// `j` is bound to the neighbor index.
#define ITER_NEIGHBORS(p_i_, origin_, inv_cell_, dims_, cell_start_, sorted_idx_, sentinel_, ...) \
    do {                                                                                          \
        const int _cx = static_cast<int>(floorf((p_i_.x - origin_.x) * inv_cell_));               \
        const int _cy = static_cast<int>(floorf((p_i_.y - origin_.y) * inv_cell_));               \
        const int _cz = static_cast<int>(floorf((p_i_.z - origin_.z) * inv_cell_));               \
        for (int dz = -1; dz <= 1; ++dz)                                                          \
        for (int dy = -1; dy <= 1; ++dy)                                                          \
        for (int dx = -1; dx <= 1; ++dx) {                                                        \
            const int nx = _cx + dx, ny = _cy + dy, nz = _cz + dz;                                \
            if (nx < 0 || nx >= dims_.x || ny < 0 || ny >= dims_.y ||                             \
                nz < 0 || nz >= dims_.z) continue;                                                \
            const u32 _c = (static_cast<u32>(nx) * static_cast<u32>(dims_.y)                      \
                          + static_cast<u32>(ny)) * static_cast<u32>(dims_.z)                     \
                          + static_cast<u32>(nz);                                                 \
            if (_c >= sentinel_) continue;                                                        \
            const u32 _begin = cell_start_[_c];                                                   \
            const u32 _end   = cell_start_[_c + 1];                                               \
            for (u32 _s = _begin; _s < _end; ++_s) {                                              \
                const u32 j = sorted_idx_[_s];                                                    \
                __VA_ARGS__;                                                                      \
            }                                                                                     \
        }                                                                                         \
    } while (0)

// ---------- boundary mass calibration (Akinci 2012) ----------
// For each boundary particle b: mass_b = rho_0 / sum_{b'} W(|x_b - x_b'|, l)
// (sum over boundary neighbors INCLUDING self via W(0)).
__global__ void compute_boundary_mass_kernel(
        const Vec3f* pos_b,
        const u32* sorted_b, const u32* cell_start_b,
        f32* mass_b_out,
        std::size_t n_boundary,
        Vec3f origin_b, Vec3i dims_b, f32 cell_length_b,
        f32 rest_density, f32 smoothing_length) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_boundary) return;

    const Vec3f p_i = pos_b[i];
    const f32 inv_cell = 1.0f / cell_length_b;
    const u32 sentinel = static_cast<u32>(dims_b.x) * static_cast<u32>(dims_b.y)
                       * static_cast<u32>(dims_b.z);

    f32 sum_w = 0.0f;
    ITER_NEIGHBORS(p_i, origin_b, inv_cell, dims_b, cell_start_b, sorted_b, sentinel, {
        const Vec3f p_j = pos_b[j];
        const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
        const f32 d = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        sum_w += sph::cubic_spline_W(d, smoothing_length);
    });

    mass_b_out[i] = rest_density / fmaxf(sum_w, 1e-9f);
}

// ---------- density (fluid + boundary contributions) ----------
__global__ void density_kernel(
        const Vec3f* pos_f,
        const u32* sorted_f, const u32* cell_start_f,
        f32* density_out,
        std::size_t n_fluid,
        Vec3f origin_f, Vec3i dims_f, f32 cell_length_f,
        f32 mass_f, f32 smoothing_length,
        // boundary (may be null/zero)
        const Vec3f* pos_b, const u32* sorted_b, const u32* cell_start_b,
        const f32* mass_b,
        Vec3f origin_b, Vec3i dims_b, f32 cell_length_b,
        std::size_t n_boundary) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fluid) return;

    const Vec3f p_i = pos_f[i];
    f32 rho = 0.0f;

    // Fluid neighbors
    {
        const f32 inv_cell = 1.0f / cell_length_f;
        const u32 sentinel = static_cast<u32>(dims_f.x) * static_cast<u32>(dims_f.y)
                           * static_cast<u32>(dims_f.z);
        ITER_NEIGHBORS(p_i, origin_f, inv_cell, dims_f, cell_start_f, sorted_f, sentinel, {
            const Vec3f p_j = pos_f[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const f32 d = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
            rho += mass_f * sph::cubic_spline_W(d, smoothing_length);
        });
    }
    // Boundary neighbors
    if (n_boundary > 0 && pos_b != nullptr) {
        const f32 inv_cell = 1.0f / cell_length_b;
        const u32 sentinel = static_cast<u32>(dims_b.x) * static_cast<u32>(dims_b.y)
                           * static_cast<u32>(dims_b.z);
        ITER_NEIGHBORS(p_i, origin_b, inv_cell, dims_b, cell_start_b, sorted_b, sentinel, {
            const Vec3f p_j = pos_b[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const f32 d = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
            rho += mass_b[j] * sph::cubic_spline_W(d, smoothing_length);
        });
    }
    density_out[i] = rho;
}

// ---------- alpha factor (Bender-Koschier eq. 11) ----------
// alpha_i = density_i / (|sum_j m_j ∇W|^2 + sum_j |m_j ∇W|^2)
// summed over BOTH fluid and boundary neighbors.
__global__ void alpha_kernel(
        const Vec3f* pos_f, const f32* density,
        const u32* sorted_f, const u32* cell_start_f,
        f32* alpha_out,
        std::size_t n_fluid,
        Vec3f origin_f, Vec3i dims_f, f32 cell_length_f,
        f32 mass_f, f32 smoothing_length,
        const Vec3f* pos_b, const u32* sorted_b, const u32* cell_start_b,
        const f32* mass_b,
        Vec3f origin_b, Vec3i dims_b, f32 cell_length_b,
        std::size_t n_boundary) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fluid) return;

    const Vec3f p_i = pos_f[i];
    Vec3f sum_grad{0.f, 0.f, 0.f};
    f32   sum_grad_sq = 0.0f;

    // Fluid neighbors (skip self)
    {
        const f32 inv_cell = 1.0f / cell_length_f;
        const u32 sentinel = static_cast<u32>(dims_f.x) * static_cast<u32>(dims_f.y)
                           * static_cast<u32>(dims_f.z);
        ITER_NEIGHBORS(p_i, origin_f, inv_cell, dims_f, cell_start_f, sorted_f, sentinel, {
            if (j == i) continue;
            const Vec3f p_j = pos_f[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const Vec3f mgW{mass_f * gW.x, mass_f * gW.y, mass_f * gW.z};
            sum_grad.x += mgW.x; sum_grad.y += mgW.y; sum_grad.z += mgW.z;
            sum_grad_sq += mgW.x * mgW.x + mgW.y * mgW.y + mgW.z * mgW.z;
        });
    }
    // Boundary neighbors
    if (n_boundary > 0 && pos_b != nullptr) {
        const f32 inv_cell = 1.0f / cell_length_b;
        const u32 sentinel = static_cast<u32>(dims_b.x) * static_cast<u32>(dims_b.y)
                           * static_cast<u32>(dims_b.z);
        ITER_NEIGHBORS(p_i, origin_b, inv_cell, dims_b, cell_start_b, sorted_b, sentinel, {
            const Vec3f p_j = pos_b[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const Vec3f mgW{mass_b[j] * gW.x, mass_b[j] * gW.y, mass_b[j] * gW.z};
            sum_grad.x += mgW.x; sum_grad.y += mgW.y; sum_grad.z += mgW.z;
            sum_grad_sq += mgW.x * mgW.x + mgW.y * mgW.y + mgW.z * mgW.z;
        });
    }

    const f32 denom = sum_grad.x * sum_grad.x + sum_grad.y * sum_grad.y
                    + sum_grad.z * sum_grad.z + sum_grad_sq;
    alpha_out[i] = (denom > 1e-9f) ? (density[i] / denom) : 0.0f;
}

// ---------- predict density change + compute kappa ----------
// rho_dot_i = sum_j m_j (v_i - v_j) . grad_W_ij
// (boundary particles have velocity 0, so they contribute m_b (v_i - 0) . gradW)
// rho_pred_i = density_i + dt * rho_dot
// kappa_i = max(0, (rho_pred - rho_0)) * alpha_i / dt^2
__global__ void density_change_kernel(
        const Vec3f* pos_f, const Vec3f* vel_f,
        const f32* density,
        const u32* sorted_f, const u32* cell_start_f,
        f32* density_adv_out, f32* kappa_out,
        const f32* alpha,
        f32 rest_density,
        std::size_t n_fluid,
        Vec3f origin_f, Vec3i dims_f, f32 cell_length_f,
        f32 mass_f, f32 smoothing_length, f32 dt,
        const Vec3f* pos_b, const u32* sorted_b, const u32* cell_start_b,
        const f32* mass_b,
        Vec3f origin_b, Vec3i dims_b, f32 cell_length_b,
        std::size_t n_boundary) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fluid) return;

    const Vec3f p_i = pos_f[i];
    const Vec3f v_i = vel_f[i];
    f32 rho_dot = 0.0f;

    // Fluid contribution
    {
        const f32 inv_cell = 1.0f / cell_length_f;
        const u32 sentinel = static_cast<u32>(dims_f.x) * static_cast<u32>(dims_f.y)
                           * static_cast<u32>(dims_f.z);
        ITER_NEIGHBORS(p_i, origin_f, inv_cell, dims_f, cell_start_f, sorted_f, sentinel, {
            if (j == i) continue;
            const Vec3f p_j = pos_f[j];
            const Vec3f v_j = vel_f[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const Vec3f dv{v_i.x - v_j.x, v_i.y - v_j.y, v_i.z - v_j.z};
            rho_dot += mass_f * (dv.x * gW.x + dv.y * gW.y + dv.z * gW.z);
        });
    }
    // Boundary contribution (boundary v_j = 0, so dv = v_i)
    if (n_boundary > 0 && pos_b != nullptr) {
        const f32 inv_cell = 1.0f / cell_length_b;
        const u32 sentinel = static_cast<u32>(dims_b.x) * static_cast<u32>(dims_b.y)
                           * static_cast<u32>(dims_b.z);
        ITER_NEIGHBORS(p_i, origin_b, inv_cell, dims_b, cell_start_b, sorted_b, sentinel, {
            const Vec3f p_j = pos_b[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            rho_dot += mass_b[j] * (v_i.x * gW.x + v_i.y * gW.y + v_i.z * gW.z);
        });
    }

    const f32 rho_pred = density[i] + dt * rho_dot;
    density_adv_out[i] = rho_pred;
    // Clamp to compression-only — proper boundary forces handle "low density at wall"
    // so we don't need bipolar pressure (which destabilizes near boundaries).
    const f32 over = fmaxf(0.0f, rho_pred - rest_density);
    kappa_out[i] = over * alpha[i] / (dt * dt);
}

// ---------- apply kappa as velocity correction ----------
// v_i -= dt * sum_j m_j (kappa_i/rho_i^2 + kappa_j/rho_j^2) grad_W_ij
// Boundary contribution uses kappa_j = kappa_i (mirror) per Akinci convention.
__global__ void apply_kappa_kernel(
        const Vec3f* pos_f,
        Vec3f* vel_f,
        const f32* density,
        const f32* kappa,
        const u32* sorted_f, const u32* cell_start_f,
        std::size_t n_fluid,
        Vec3f origin_f, Vec3i dims_f, f32 cell_length_f,
        f32 mass_f, f32 smoothing_length, f32 dt,
        f32 rest_density,
        const Vec3f* pos_b, const u32* sorted_b, const u32* cell_start_b,
        const f32* mass_b,
        Vec3f origin_b, Vec3i dims_b, f32 cell_length_b,
        std::size_t n_boundary) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fluid) return;

    const Vec3f p_i = pos_f[i];
    const f32 rho_i = density[i];
    const f32 k_i   = kappa[i];
    const f32 ki_factor = k_i / fmaxf(rho_i * rho_i, 1e-9f);

    Vec3f delta_v{0.f, 0.f, 0.f};

    // Fluid contribution
    {
        const f32 inv_cell = 1.0f / cell_length_f;
        const u32 sentinel = static_cast<u32>(dims_f.x) * static_cast<u32>(dims_f.y)
                           * static_cast<u32>(dims_f.z);
        ITER_NEIGHBORS(p_i, origin_f, inv_cell, dims_f, cell_start_f, sorted_f, sentinel, {
            if (j == i) continue;
            const Vec3f p_j = pos_f[j];
            const f32 rho_j = density[j];
            const f32 k_j   = kappa[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const f32 kj_factor = k_j / fmaxf(rho_j * rho_j, 1e-9f);
            const f32 c2 = mass_f * (ki_factor + kj_factor);
            delta_v.x -= dt * c2 * gW.x;
            delta_v.y -= dt * c2 * gW.y;
            delta_v.z -= dt * c2 * gW.z;
        });
    }
    // Boundary contribution: kappa_j = kappa_i (Akinci); rho_j ≈ rest_density.
    if (n_boundary > 0 && pos_b != nullptr) {
        const f32 inv_cell = 1.0f / cell_length_b;
        const u32 sentinel = static_cast<u32>(dims_b.x) * static_cast<u32>(dims_b.y)
                           * static_cast<u32>(dims_b.z);
        const f32 boundary_factor = k_i / fmaxf(rest_density * rest_density, 1e-9f);
        ITER_NEIGHBORS(p_i, origin_b, inv_cell, dims_b, cell_start_b, sorted_b, sentinel, {
            const Vec3f p_j = pos_b[j];
            const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
            const Vec3f gW = sph::cubic_spline_grad_W(r, smoothing_length);
            const f32 c2 = mass_b[j] * (ki_factor + boundary_factor);
            delta_v.x -= dt * c2 * gW.x;
            delta_v.y -= dt * c2 * gW.y;
            delta_v.z -= dt * c2 * gW.z;
        });
    }

    vel_f[i].x += delta_v.x;
    vel_f[i].y += delta_v.y;
    vel_f[i].z += delta_v.z;
}

// ---------- XSPH viscosity (velocity smoothing) ----------
// v_i ← v_i + epsilon * sum_j (m_j / rho_avg) * (v_j - v_i) * W(r, l)
__global__ void xsph_viscosity_kernel(
        const Vec3f* pos_f, Vec3f* vel_f,
        const f32* density,
        const u32* sorted_f, const u32* cell_start_f,
        std::size_t n_fluid,
        Vec3f origin_f, Vec3i dims_f, f32 cell_length_f,
        f32 mass_f, f32 smoothing_length, f32 epsilon) {

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_fluid) return;

    const Vec3f p_i = pos_f[i];
    const Vec3f v_i = vel_f[i];
    const f32 rho_i = density[i];
    Vec3f delta{0.f, 0.f, 0.f};

    const f32 inv_cell = 1.0f / cell_length_f;
    const u32 sentinel = static_cast<u32>(dims_f.x) * static_cast<u32>(dims_f.y)
                       * static_cast<u32>(dims_f.z);
    ITER_NEIGHBORS(p_i, origin_f, inv_cell, dims_f, cell_start_f, sorted_f, sentinel, {
        if (j == i) continue;
        const Vec3f p_j = pos_f[j];
        const Vec3f v_j = vel_f[j];
        const Vec3f r{p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z};
        const f32 d = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        const f32 W = sph::cubic_spline_W(d, smoothing_length);
        const f32 rho_avg = 0.5f * (rho_i + density[j]);
        const f32 c = epsilon * (mass_f / fmaxf(rho_avg, 1e-9f)) * W;
        delta.x += c * (v_j.x - v_i.x);
        delta.y += c * (v_j.y - v_i.y);
        delta.z += c * (v_j.z - v_i.z);
    });

    vel_f[i].x += delta.x;
    vel_f[i].y += delta.y;
    vel_f[i].z += delta.z;
}

// ---------- gravity, advection ----------
__global__ void apply_gravity_kernel(Vec3f* velocity, std::size_t n,
                                      Vec3f g, f32 dt) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    velocity[i].x += g.x * dt;
    velocity[i].y += g.y * dt;
    velocity[i].z += g.z * dt;
}

// Pure advection — no AABB clamp. Boundary particles handle wall reactions.
__global__ void advect_kernel(Vec3f* position, Vec3f* velocity, std::size_t n, f32 dt) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    position[i].x += velocity[i].x * dt;
    position[i].y += velocity[i].y * dt;
    position[i].z += velocity[i].z * dt;
}

// ---------- per-particle scalars for CUB reductions ----------
__global__ void density_error_kernel(const f32* density_adv, f32 rest_density,
                                      f32* error_out, std::size_t n_particles) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    error_out[i] = fmaxf(0.0f, density_adv[i] - rest_density);
}

__global__ void velocity_magnitude_kernel(const Vec3f* v, f32* mags, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mags[i] = sqrtf(v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z);
}

#undef ITER_NEIGHBORS

} // namespace water::solvers::detail

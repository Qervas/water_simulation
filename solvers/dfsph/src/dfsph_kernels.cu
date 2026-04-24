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

} // namespace water::solvers::detail

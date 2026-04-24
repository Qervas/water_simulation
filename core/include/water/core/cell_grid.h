#pragma once

#include "water/core/device_buffer.h"
#include "water/core/types.h"
#include <cuda_runtime.h>

namespace water {

// Uniform 3D cell grid for fixed-radius SPH neighbor queries. Optimal for
// O(N) neighbor lookups when the search radius equals the cell size; the
// LBVH is used separately for ray tracing in Phase 5.
//
// Lifecycle: construct once with the scene's cell dimensions, then call
// build(positions, count) every substep before any solver kernel that
// queries neighbors.
class CellGrid {
public:
    CellGrid(Vec3f origin, Vec3i cells_per_axis, f32 cell_length,
             cudaStream_t stream = 0);

    void reserve(std::size_t capacity);
    void build(const Vec3f* positions, std::size_t count);

    const u32* cell_start()       const noexcept { return cell_start_.data(); }
    const u32* sorted_indices()   const noexcept { return sorted_indices_.data(); }
    Vec3f  origin()         const noexcept { return origin_;          }
    Vec3i  cells_per_axis() const noexcept { return cells_per_axis_;  }
    f32    cell_length()    const noexcept { return cell_length_;     }
    u32    total_cells()    const noexcept {
        return static_cast<u32>(cells_per_axis_.x)
             * static_cast<u32>(cells_per_axis_.y)
             * static_cast<u32>(cells_per_axis_.z);
    }

    cudaStream_t stream() const noexcept { return stream_; }

private:
    Vec3f         origin_;
    Vec3i         cells_per_axis_;
    f32           cell_length_;
    cudaStream_t  stream_;

    std::size_t   capacity_  = 0;
    std::size_t   count_     = 0;

    DeviceBuffer<u32>           cell_ids_;
    DeviceBuffer<u32>           sorted_indices_;
    DeviceBuffer<u32>           cell_start_;
    DeviceBuffer<unsigned char> cub_temp_;
};

} // namespace water

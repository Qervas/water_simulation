#pragma once

#include "water/core/device_buffer.h"
#include "water/core/types.h"
#include <cuda_runtime.h>

namespace water {

struct AABB {
    Vec3f min;
    Vec3f max;
};

// Linear BVH built from a particle position array. After build(), neighbor
// queries can be issued from device kernels via the `query` kernel helper
// declared in spatial_accel_device.cuh (out of scope for Phase 1 — Phase 2
// will provide this; for now, build() correctness is verified via host-side
// readback of node AABBs and sorted indices).
class SpatialAccel {
public:
    SpatialAccel() = default;

    // Build the BVH for `count` positions. The internal node array has
    // (count - 1) entries; the leaf array has `count` entries; each leaf
    // wraps a sorted-particle index.
    void build(const Vec3f* positions, std::size_t count, AABB scene_bounds,
               cudaStream_t stream = 0);

    std::size_t leaf_count() const noexcept { return leaf_count_; }

    // Sorted permutation: sorted_indices_[i] == original particle index that
    // occupies sorted slot i. Useful for sorting per-particle data.
    const u32* sorted_indices() const noexcept { return sorted_indices_.data(); }

    const u32* internal_left()  const noexcept { return internal_left_.data();  }
    const u32* internal_right() const noexcept { return internal_right_.data(); }
    const AABB* node_aabbs()    const noexcept { return node_aabbs_.data();     }

private:
    std::size_t leaf_count_ = 0;

    // Per-leaf (size = leaf_count_)
    DeviceBuffer<u32>   morton_codes_;
    DeviceBuffer<u32>   sorted_indices_;

    // Internal nodes (size = leaf_count_ - 1)
    DeviceBuffer<u32>   internal_left_;
    DeviceBuffer<u32>   internal_right_;
    DeviceBuffer<u32>   internal_parent_;

    // AABBs for both internal (first leaf_count_-1) and leaves (next leaf_count_)
    DeviceBuffer<AABB>  node_aabbs_;
};

} // namespace water

#include "water/core/spatial_accel.h"
#include "water/core/cuda_check.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace water {

namespace {

// Expand a 10-bit integer into 30 bits by inserting two zeros between each bit.
__device__ inline u32 expand_bits_10(u32 v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Morton code from a position normalized to [0, 1]^3.
__device__ inline u32 morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    u32 xx = expand_bits_10(static_cast<u32>(x));
    u32 yy = expand_bits_10(static_cast<u32>(y));
    u32 zz = expand_bits_10(static_cast<u32>(z));
    return (xx << 2) | (yy << 1) | zz;
}

__global__ void compute_morton(const Vec3f* positions, u32* codes, u32* indices,
                                std::size_t n, Vec3f scene_min, Vec3f scene_extent_inv) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3f p = positions[i];
    float nx = (p.x - scene_min.x) * scene_extent_inv.x;
    float ny = (p.y - scene_min.y) * scene_extent_inv.y;
    float nz = (p.z - scene_min.z) * scene_extent_inv.z;
    codes[i] = morton3D(nx, ny, nz);
    indices[i] = static_cast<u32>(i);
}

// Karras 2012: count leading zeros of (codes[i] XOR codes[j]); ties broken by
// using the 32-bit indices i and j to ensure a unique total order.
__device__ inline int delta(int i, int j, const u32* codes, int n) {
    if (j < 0 || j >= n) return -1;
    u32 ci = codes[i];
    u32 cj = codes[j];
    if (ci == cj) {
        // Use the indices themselves as a tie-breaker.
        return 32 + __clz(static_cast<u32>(i ^ j));
    }
    return __clz(ci ^ cj);
}

__global__ void build_internal_nodes(const u32* codes, u32* lefts, u32* rights,
                                      u32* parents, int n_leaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_internal = n_leaves - 1;
    if (i >= n_internal) return;

    // Determine direction of the range (+1 or -1) using delta sign.
    int d = (delta(i, i + 1, codes, n_leaves) - delta(i, i - 1, codes, n_leaves)) > 0
            ? 1 : -1;

    // Compute upper bound for the length of the range.
    int delta_min = delta(i, i - d, codes, n_leaves);
    int l_max = 2;
    while (delta(i, i + l_max * d, codes, n_leaves) > delta_min) l_max *= 2;

    // Find exact end of range using binary search.
    int l = 0;
    for (int t = l_max / 2; t > 0; t /= 2) {
        if (delta(i, i + (l + t) * d, codes, n_leaves) > delta_min) l += t;
    }
    int j = i + l * d;

    // Find the split position using binary search.
    int delta_node = delta(i, j, codes, n_leaves);
    int s = 0;
    int t = (l + 1) / 2;
    while (t > 0) {
        if (delta(i, i + (s + t) * d, codes, n_leaves) > delta_node) s += t;
        t = (t == 1) ? 0 : (t + 1) / 2;
    }
    int gamma = i + s * d + min(d, 0);

    // Assign children.
    int left  = (min(i, j) == gamma)     ? (n_internal + gamma)     : gamma;
    int right = (max(i, j) == gamma + 1) ? (n_internal + gamma + 1) : (gamma + 1);

    lefts[i]  = static_cast<u32>(left);
    rights[i] = static_cast<u32>(right);
    parents[left]  = static_cast<u32>(i);
    parents[right] = static_cast<u32>(i);
}

__global__ void init_leaf_aabbs(const Vec3f* positions, const u32* sorted_indices,
                                 AABB* node_aabbs, int n_internal, int n_leaves,
                                 float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_leaves) return;
    Vec3f p = positions[sorted_indices[i]];
    AABB box;
    box.min = {p.x - radius, p.y - radius, p.z - radius};
    box.max = {p.x + radius, p.y + radius, p.z + radius};
    node_aabbs[n_internal + i] = box;
}

} // namespace

void SpatialAccel::build(const Vec3f* positions, std::size_t count,
                          AABB scene_bounds, cudaStream_t stream) {
    if (count < 2) {
        // Degenerate cases: 0 or 1 particles. Allocate the leaf storage
        // (1 slot) but skip the internal-node tree construction entirely.
        leaf_count_ = count;
        if (count == 1) {
            sorted_indices_ = DeviceBuffer<u32>(1, stream);
            node_aabbs_     = DeviceBuffer<AABB>(1, stream);
            // Fill sorted_indices[0] = 0
            u32 zero = 0;
            WATER_CUDA_CHECK(cudaMemcpyAsync(sorted_indices_.data(), &zero,
                                              sizeof(u32),
                                              cudaMemcpyHostToDevice, stream));
        }
        return;
    }

    leaf_count_ = count;
    int n_internal = static_cast<int>(count) - 1;

    morton_codes_    = DeviceBuffer<u32>(count, stream);
    sorted_indices_  = DeviceBuffer<u32>(count, stream);
    internal_left_   = DeviceBuffer<u32>(n_internal, stream);
    internal_right_  = DeviceBuffer<u32>(n_internal, stream);
    internal_parent_ = DeviceBuffer<u32>(count + n_internal, stream);
    node_aabbs_      = DeviceBuffer<AABB>(count + n_internal, stream);

    Vec3f extent = {
        scene_bounds.max.x - scene_bounds.min.x,
        scene_bounds.max.y - scene_bounds.min.y,
        scene_bounds.max.z - scene_bounds.min.z,
    };
    Vec3f extent_inv = {
        extent.x > 0 ? 1.0f / extent.x : 0.0f,
        extent.y > 0 ? 1.0f / extent.y : 0.0f,
        extent.z > 0 ? 1.0f / extent.z : 0.0f,
    };

    int block = 256;
    int grid  = static_cast<int>((count + block - 1) / block);
    compute_morton<<<grid, block, 0, stream>>>(positions,
                                                morton_codes_.data(),
                                                sorted_indices_.data(),
                                                count, scene_bounds.min, extent_inv);
    WATER_CUDA_CHECK_LAST();

    // Sort indices by morton key using CUB device-wide radix sort.
    DeviceBuffer<u32> sorted_codes(count, stream);
    DeviceBuffer<u32> sorted_idx_out(count, stream);

    std::size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        morton_codes_.data(), sorted_codes.data(),
        sorted_indices_.data(), sorted_idx_out.data(),
        static_cast<int>(count), 0, 30, stream);
    DeviceBuffer<unsigned char> temp_storage(temp_bytes, stream);
    cub::DeviceRadixSort::SortPairs(temp_storage.data(), temp_bytes,
        morton_codes_.data(), sorted_codes.data(),
        sorted_indices_.data(), sorted_idx_out.data(),
        static_cast<int>(count), 0, 30, stream);

    // Move sorted results back into our owned buffers.
    WATER_CUDA_CHECK(cudaMemcpyAsync(morton_codes_.data(), sorted_codes.data(),
                                      count * sizeof(u32),
                                      cudaMemcpyDeviceToDevice, stream));
    WATER_CUDA_CHECK(cudaMemcpyAsync(sorted_indices_.data(), sorted_idx_out.data(),
                                      count * sizeof(u32),
                                      cudaMemcpyDeviceToDevice, stream));

    grid = (n_internal + block - 1) / block;
    build_internal_nodes<<<grid, block, 0, stream>>>(
        morton_codes_.data(), internal_left_.data(), internal_right_.data(),
        internal_parent_.data(), static_cast<int>(count));
    WATER_CUDA_CHECK_LAST();

    // Initialize leaf AABBs (radius=0; expand later via solver-supplied radius).
    grid = (static_cast<int>(count) + block - 1) / block;
    init_leaf_aabbs<<<grid, block, 0, stream>>>(
        positions, sorted_indices_.data(), node_aabbs_.data(),
        n_internal, static_cast<int>(count), 0.0f);
    WATER_CUDA_CHECK_LAST();

    // NOTE: internal-node AABB propagation is intentionally deferred to
    // Phase 2. Phase 1 tests verify only the sorted-index permutation, leaf
    // AABBs, and that internal-node child indices form a valid tree. The
    // bottom-up AABB merge pass is implemented when first solver kernel
    // needs it.
}

} // namespace water

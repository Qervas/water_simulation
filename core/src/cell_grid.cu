#include "water/core/cell_grid.h"
#include "water/core/cuda_check.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace water {

namespace {

__device__ inline u32 hash_cell(Vec3f p, Vec3f origin, f32 inv_cell, Vec3i dims) {
    int cx = static_cast<int>(floorf((p.x - origin.x) * inv_cell));
    int cy = static_cast<int>(floorf((p.y - origin.y) * inv_cell));
    int cz = static_cast<int>(floorf((p.z - origin.z) * inv_cell));
    if (cx < 0 || cx >= dims.x ||
        cy < 0 || cy >= dims.y ||
        cz < 0 || cz >= dims.z) {
        return static_cast<u32>(dims.x) * static_cast<u32>(dims.y)
             * static_cast<u32>(dims.z);
    }
    return (static_cast<u32>(cx) * static_cast<u32>(dims.y) + static_cast<u32>(cy))
         * static_cast<u32>(dims.z) + static_cast<u32>(cz);
}

__global__ void compute_cell_ids(const Vec3f* positions, u32* cell_ids, u32* indices,
                                  std::size_t n, Vec3f origin, f32 inv_cell, Vec3i dims) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cell_ids[i] = hash_cell(positions[i], origin, inv_cell, dims);
    indices[i] = static_cast<u32>(i);
}

__global__ void fill_u32(u32* p, u32 v, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void scatter_starts(const u32* sorted_cell_ids, u32* cell_start, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 c = sorted_cell_ids[i];
    if (i == 0) {
        cell_start[c] = 0;
    } else {
        u32 prev = sorted_cell_ids[i - 1];
        if (prev != c) cell_start[c] = static_cast<u32>(i);
    }
}

__global__ void propagate_starts_right_to_left(u32* cell_start, std::size_t total_plus_one) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(total_plus_one) - 2; i >= 0; --i) {
        if (cell_start[i] > cell_start[i + 1]) cell_start[i] = cell_start[i + 1];
    }
}

} // namespace

CellGrid::CellGrid(Vec3f origin, Vec3i cells_per_axis, f32 cell_length,
                    cudaStream_t stream)
    : origin_(origin), cells_per_axis_(cells_per_axis),
      cell_length_(cell_length), stream_(stream),
      cell_start_(static_cast<std::size_t>(total_cells()) + 1, stream) {
    cell_start_.fill_zero();
}

void CellGrid::reserve(std::size_t capacity) {
    if (capacity <= capacity_) return;
    capacity_       = capacity;
    cell_ids_       = DeviceBuffer<u32>(capacity, stream_);
    sorted_indices_ = DeviceBuffer<u32>(capacity, stream_);
}

void CellGrid::build(const Vec3f* positions, std::size_t count) {
    if (count == 0) {
        count_ = 0;
        return;
    }
    if (count > capacity_) reserve(count);
    count_ = count;

    const f32 inv_cell = 1.0f / cell_length_;
    constexpr int block = 256;

    // 1. compute (cell_id, index) per particle
    {
        const int grid = static_cast<int>((count + block - 1) / block);
        compute_cell_ids<<<grid, block, 0, stream_>>>(
            positions, cell_ids_.data(), sorted_indices_.data(),
            count, origin_, inv_cell, cells_per_axis_);
        WATER_CUDA_CHECK_LAST();
    }

    // 2. sort by cell_id with CUB DeviceRadixSort
    DeviceBuffer<u32> sorted_keys(count, stream_);
    DeviceBuffer<u32> sorted_vals(count, stream_);
    {
        std::size_t bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            cell_ids_.data(), sorted_keys.data(),
            sorted_indices_.data(), sorted_vals.data(),
            static_cast<int>(count), 0, sizeof(u32) * 8, stream_);
        if (bytes > cub_temp_.size()) {
            cub_temp_ = DeviceBuffer<unsigned char>(bytes, stream_);
        }
        cub::DeviceRadixSort::SortPairs(
            cub_temp_.data(), bytes,
            cell_ids_.data(), sorted_keys.data(),
            sorted_indices_.data(), sorted_vals.data(),
            static_cast<int>(count), 0, sizeof(u32) * 8, stream_);
        WATER_CUDA_CHECK(cudaMemcpyAsync(cell_ids_.data(), sorted_keys.data(),
                                          count * sizeof(u32),
                                          cudaMemcpyDeviceToDevice, stream_));
        WATER_CUDA_CHECK(cudaMemcpyAsync(sorted_indices_.data(), sorted_vals.data(),
                                          count * sizeof(u32),
                                          cudaMemcpyDeviceToDevice, stream_));
    }

    // 3. compute cell_start[]: fill with sentinel, scatter at boundaries,
    //    right-to-left min-scan to fill empty-cell sentinels.
    const u32 sentinel = static_cast<u32>(count);
    const std::size_t cs_count = static_cast<std::size_t>(total_cells()) + 1;
    {
        const int g = static_cast<int>((cs_count + block - 1) / block);
        fill_u32<<<g, block, 0, stream_>>>(cell_start_.data(), sentinel, cs_count);
        WATER_CUDA_CHECK_LAST();
    }
    {
        const int g = static_cast<int>((count + block - 1) / block);
        scatter_starts<<<g, block, 0, stream_>>>(cell_ids_.data(), cell_start_.data(), count);
        WATER_CUDA_CHECK_LAST();
    }
    {
        propagate_starts_right_to_left<<<1, 1, 0, stream_>>>(
            cell_start_.data(), cs_count);
        WATER_CUDA_CHECK_LAST();
    }
}

} // namespace water

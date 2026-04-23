#include <doctest/doctest.h>
#include "water/core/spatial_accel.h"
#include "water/core/cuda_check.h"
#include <vector>
#include <algorithm>

using namespace water;

TEST_CASE("SpatialAccel: build on 8 grid points produces sorted indices") {
    // 2x2x2 grid in unit cube
    std::vector<Vec3f> positions = {
        {0.1f, 0.1f, 0.1f}, {0.9f, 0.1f, 0.1f},
        {0.1f, 0.9f, 0.1f}, {0.9f, 0.9f, 0.1f},
        {0.1f, 0.1f, 0.9f}, {0.9f, 0.1f, 0.9f},
        {0.1f, 0.9f, 0.9f}, {0.9f, 0.9f, 0.9f},
    };

    Vec3f* d_pos = nullptr;
    WATER_CUDA_CHECK(cudaMalloc(&d_pos, sizeof(Vec3f) * positions.size()));
    WATER_CUDA_CHECK(cudaMemcpy(d_pos, positions.data(),
                                 sizeof(Vec3f) * positions.size(),
                                 cudaMemcpyHostToDevice));

    SpatialAccel accel;
    AABB bounds{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    accel.build(d_pos, positions.size(), bounds);
    cudaDeviceSynchronize();

    CHECK(accel.leaf_count() == 8);

    std::vector<u32> sorted(8);
    WATER_CUDA_CHECK(cudaMemcpy(sorted.data(), accel.sorted_indices(),
                                 sizeof(u32) * 8, cudaMemcpyDeviceToHost));

    // The sorted indices must be a permutation of [0, 8).
    std::vector<u32> sorted_copy = sorted;
    std::sort(sorted_copy.begin(), sorted_copy.end());
    for (u32 i = 0; i < 8; ++i) CHECK(sorted_copy[i] == i);

    cudaFree(d_pos);
}

TEST_CASE("SpatialAccel: degenerate single-particle build does not crash") {
    Vec3f pos{0.5f, 0.5f, 0.5f};
    Vec3f* d_pos = nullptr;
    WATER_CUDA_CHECK(cudaMalloc(&d_pos, sizeof(Vec3f)));
    WATER_CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(Vec3f),
                                 cudaMemcpyHostToDevice));
    SpatialAccel accel;
    AABB bounds{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    accel.build(d_pos, 1, bounds);
    cudaDeviceSynchronize();
    CHECK(accel.leaf_count() == 1);
    cudaFree(d_pos);
}

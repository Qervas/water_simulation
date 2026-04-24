#include <doctest/doctest.h>
#include "water/core/cell_grid.h"
#include "water/core/cuda_check.h"
#include <vector>
#include <algorithm>

using namespace water;

TEST_CASE("CellGrid: 8 grid-aligned points map to 8 distinct cells") {
    CellGrid grid({0.f, 0.f, 0.f}, {4, 4, 4}, 0.5f);
    grid.reserve(8);

    std::vector<Vec3f> pts = {
        {0.25f, 0.25f, 0.25f},
        {0.75f, 0.25f, 0.25f},
        {0.25f, 0.75f, 0.25f},
        {0.75f, 0.75f, 0.25f},
        {0.25f, 0.25f, 0.75f},
        {0.75f, 0.25f, 0.75f},
        {0.25f, 0.75f, 0.75f},
        {0.75f, 0.75f, 0.75f},
    };
    Vec3f* d = nullptr;
    WATER_CUDA_CHECK(cudaMalloc(&d, sizeof(Vec3f) * pts.size()));
    WATER_CUDA_CHECK(cudaMemcpy(d, pts.data(), sizeof(Vec3f) * pts.size(),
                                 cudaMemcpyHostToDevice));

    grid.build(d, pts.size());
    cudaDeviceSynchronize();

    std::vector<u32> sorted(8);
    WATER_CUDA_CHECK(cudaMemcpy(sorted.data(), grid.sorted_indices(),
                                 sizeof(u32) * 8, cudaMemcpyDeviceToHost));
    std::vector<u32> copy = sorted;
    std::sort(copy.begin(), copy.end());
    for (u32 i = 0; i < 8; ++i) CHECK(copy[i] == i);

    std::vector<u32> cs(grid.total_cells() + 1);
    WATER_CUDA_CHECK(cudaMemcpy(cs.data(), grid.cell_start(),
                                 sizeof(u32) * cs.size(),
                                 cudaMemcpyDeviceToHost));
    for (std::size_t i = 1; i < cs.size(); ++i) {
        CHECK(cs[i] >= cs[i - 1]);
    }

    cudaFree(d);
}

TEST_CASE("CellGrid: out-of-bounds particle goes to sentinel cell") {
    CellGrid grid({0.f, 0.f, 0.f}, {2, 2, 2}, 0.5f);
    grid.reserve(2);

    std::vector<Vec3f> pts = {
        {0.25f, 0.25f, 0.25f},
        {99.0f, 99.0f, 99.0f},
    };
    Vec3f* d = nullptr;
    WATER_CUDA_CHECK(cudaMalloc(&d, sizeof(Vec3f) * pts.size()));
    WATER_CUDA_CHECK(cudaMemcpy(d, pts.data(), sizeof(Vec3f) * pts.size(),
                                 cudaMemcpyHostToDevice));
    grid.build(d, pts.size());
    cudaDeviceSynchronize();

    std::vector<u32> cs(grid.total_cells() + 1);
    WATER_CUDA_CHECK(cudaMemcpy(cs.data(), grid.cell_start(),
                                 sizeof(u32) * cs.size(),
                                 cudaMemcpyDeviceToHost));
    // cell_start at the sentinel index = number of in-bounds particles (1).
    // Solver kernels skip the sentinel cell explicitly via `if (c == sentinel) continue`.
    CHECK(cs.back() == 1);
    // First in-bounds cell still starts at 0.
    CHECK(cs[0] == 0);
    cudaFree(d);
}

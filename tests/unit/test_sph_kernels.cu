#include <doctest/doctest.h>
#include "water/core/sph_kernels.cuh"
#include "water/core/cuda_check.h"
#include <cmath>

using namespace water;

namespace {

__global__ void eval_W(f32 r, f32 l, f32* out) { *out = sph::cubic_spline_W(r, l); }
__global__ void eval_grad_W(Vec3f r_ij, f32 l, Vec3f* out) {
    *out = sph::cubic_spline_grad_W(r_ij, l);
}

f32 host_eval_W(f32 r, f32 l) {
    f32* d; f32 h = 0.f;
    cudaMalloc(&d, sizeof(f32));
    eval_W<<<1, 1>>>(r, l, d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(f32), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}
Vec3f host_eval_grad_W(Vec3f r_ij, f32 l) {
    Vec3f* d; Vec3f h{};
    cudaMalloc(&d, sizeof(Vec3f));
    eval_grad_W<<<1, 1>>>(r_ij, l, d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}

} // namespace

TEST_CASE("SPH kernel: W(r=0, l) is the peak; W(r>=2l) is zero") {
    const f32 l = 0.05f;
    CHECK(host_eval_W(0.0f, l) > 0.0f);
    CHECK(host_eval_W(2.0f * l, l) == doctest::Approx(0.0f).epsilon(1e-6f));
    CHECK(host_eval_W(3.0f * l, l) == doctest::Approx(0.0f));
}

TEST_CASE("SPH kernel: W is monotonically non-increasing on [0, 2l]") {
    const f32 l = 0.1f;
    f32 prev = host_eval_W(0.0f, l);
    for (int i = 1; i <= 20; ++i) {
        const f32 r = i * 0.1f * l;
        f32 cur = host_eval_W(r, l);
        CHECK(cur <= prev + 1e-7f);
        prev = cur;
    }
}

TEST_CASE("SPH kernel: ∇W is zero at r=0 and at r >= 2l") {
    const f32 l = 0.05f;
    auto g0 = host_eval_grad_W({0.0f, 0.0f, 0.0f}, l);
    CHECK(g0.x == doctest::Approx(0.0f));
    CHECK(g0.y == doctest::Approx(0.0f));
    CHECK(g0.z == doctest::Approx(0.0f));

    auto gOut = host_eval_grad_W({2.5f * l, 0.0f, 0.0f}, l);
    CHECK(gOut.x == doctest::Approx(0.0f));
}

TEST_CASE("SPH kernel: ∇W(r_ij) points opposite r_ij inside support (dW/dq < 0)") {
    const f32 l = 0.05f;
    auto g = host_eval_grad_W({0.5f * l, 0.0f, 0.0f}, l);
    CHECK(g.x < 0.0f);
    CHECK(g.y == doctest::Approx(0.0f));
    CHECK(g.z == doctest::Approx(0.0f));
}

#include <doctest/doctest.h>
#include "water/core/particle_store.h"
#include "water/core/cuda_check.h"
#include <vector>

using namespace water;

TEST_CASE("ParticleStore: construction registers position+velocity") {
    ParticleStore store(100);

    CHECK(store.capacity() == 100);
    CHECK(store.count() == 0);
    CHECK(store.has_attribute("position"));
    CHECK(store.has_attribute("velocity"));
    CHECK_FALSE(store.has_attribute("density"));
}

TEST_CASE("ParticleStore: resize within capacity OK, beyond throws") {
    ParticleStore store(50);

    store.resize(10);
    CHECK(store.count() == 10);

    store.resize(50);
    CHECK(store.count() == 50);

    CHECK_THROWS_AS(store.resize(51), std::out_of_range);
}

TEST_CASE("ParticleStore: register custom attribute") {
    ParticleStore store(64);
    auto density_h = store.register_attribute<float>("density", AttribType::F32);

    CHECK(density_h.valid());
    CHECK(store.has_attribute("density"));

    // Duplicate registration throws.
    CHECK_THROWS_AS(
        store.register_attribute<float>("density", AttribType::F32),
        std::runtime_error);
}

TEST_CASE("ParticleStore: position attribute is writable from host then readable") {
    ParticleStore store(4);
    store.resize(4);

    std::vector<Vec3f> host_positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };

    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), host_positions.data(),
                                 sizeof(Vec3f) * 4, cudaMemcpyHostToDevice));

    std::vector<Vec3f> readback(4);
    WATER_CUDA_CHECK(cudaMemcpy(readback.data(), store.positions(),
                                 sizeof(Vec3f) * 4, cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(readback[i].x == doctest::Approx(host_positions[i].x));
        CHECK(readback[i].y == doctest::Approx(host_positions[i].y));
        CHECK(readback[i].z == doctest::Approx(host_positions[i].z));
    }
}

TEST_CASE("ParticleStore: custom F32 attribute round-trip") {
    ParticleStore store(8);
    store.resize(8);
    auto h = store.register_attribute<float>("temperature", AttribType::F32);

    std::vector<float> host = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    WATER_CUDA_CHECK(cudaMemcpy(store.attribute_data(h), host.data(),
                                 sizeof(float) * 8, cudaMemcpyHostToDevice));

    std::vector<float> readback(8);
    WATER_CUDA_CHECK(cudaMemcpy(readback.data(), store.attribute_data(h),
                                 sizeof(float) * 8, cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < 8; ++i)
        CHECK(readback[i] == doctest::Approx(host[i]));
}

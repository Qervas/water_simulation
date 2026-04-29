#include <doctest/doctest.h>
#include "water/solvers/boundary_sampler.h"

using namespace water;
using namespace water::solvers;

TEST_CASE("boundary sampler: counts roughly match 6 * n_layers * area / spacing^2") {
    auto pts = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.05f, 1);
    // 6 faces * 1 layer * (1/0.05+1)^2 = 6 * 441 = 2646 with edge double-counts
    CHECK(pts.size() > 2000);
    CHECK(pts.size() < 4000);
}

TEST_CASE("boundary sampler: every point lies on or just inside a face") {
    auto pts = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.05f, 2);
    const f32 max_inward = 0.05f * 2.0f + 1e-3f;  // 2 layers
    for (auto p : pts) {
        const bool near_face =
            (p.x <= max_inward) || (p.x >= 1.0f - max_inward) ||
            (p.y <= max_inward) || (p.y >= 1.0f - max_inward) ||
            (p.z <= max_inward) || (p.z >= 1.0f - max_inward);
        CHECK(near_face);
    }
}

TEST_CASE("boundary sampler: 2 layers produces ~2x as many points as 1 layer") {
    auto p1 = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.05f, 1);
    auto p2 = sample_aabb_boundary({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0.05f, 2);
    CHECK(p2.size() > p1.size() * 1.5);
    CHECK(p2.size() < p1.size() * 2.5);
}

#include <doctest/doctest.h>
#include "water/core/boundary.h"

using namespace water;

TEST_CASE("Boundary: default unit cube") {
    Boundary b;
    CHECK(b.min().x == doctest::Approx(0.0f));
    CHECK(b.max().z == doctest::Approx(1.0f));
}

TEST_CASE("Boundary: explicit min/max") {
    Boundary b({-1.0f, -2.0f, -3.0f}, {1.0f, 2.0f, 3.0f});
    CHECK(b.min().x == doctest::Approx(-1.0f));
    CHECK(b.max().y == doctest::Approx(2.0f));
}

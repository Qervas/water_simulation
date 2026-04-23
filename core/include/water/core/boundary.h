#pragma once

#include "water/core/types.h"

namespace water {

// Phase 1 stub. Phase 2 replaces this with an SDF-sampled boundary.
class Boundary {
public:
    Boundary() = default;

    // For now: a simple AABB box collider (interior). Particles outside the
    // box get pushed back along the violated axis.
    explicit Boundary(Vec3f min, Vec3f max) : min_(min), max_(max) {}

    Vec3f min() const noexcept { return min_; }
    Vec3f max() const noexcept { return max_; }

private:
    Vec3f min_{0.0f, 0.0f, 0.0f};
    Vec3f max_{1.0f, 1.0f, 1.0f};
};

} // namespace water

#pragma once

#include "water/core/types.h"
#include <vector>

namespace water::solvers {

// Sample the interior surfaces of an AABB box with regularly spaced static
// SPH particles. n_layers > 1 gives multiple inward layers (better kernel
// support for fluid neighbors near the wall).
//
// For a box [lo, hi] with `spacing` between particles and `n_layers` layers
// per face, total particles ≈ 6 * n_layers * area / spacing².
//
// Note: edges/corners are emitted as overlapping points from adjacent faces;
// the Akinci 2012 mass calibration (compute_boundary_mass) accounts for
// this naturally — no need to deduplicate.
std::vector<Vec3f> sample_aabb_boundary(
    Vec3f lo, Vec3f hi, f32 spacing, int n_layers = 2);

} // namespace water::solvers

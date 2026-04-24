#pragma once

#include "water/core/types.h"
#include <cuda_runtime.h>

namespace water::sph {

__device__ inline f32 pi_f() { return 3.14159265358979323846f; }

// Cubic spline kernel W(r, l), Monaghan's 3D variant.
// Convention: smoothing length 'l' has support [0, 2l] (q = r/l, W=0 for q>=2).
__device__ inline f32 cubic_spline_W(f32 r, f32 l) {
    const f32 q = r / l;
    if (q >= 2.0f) return 0.0f;
    const f32 sigma = 1.0f / (pi_f() * l * l * l);
    if (q < 1.0f) {
        return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    } else {
        const f32 t = 2.0f - q;
        return sigma * 0.25f * t * t * t;
    }
}

// Gradient ∇_i W(r_ij, l) where r_ij = x_i - x_j.
__device__ inline Vec3f cubic_spline_grad_W(Vec3f r_ij, f32 l) {
    const f32 r2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
    const f32 r  = sqrtf(r2);
    if (r < 1e-12f) return {0.f, 0.f, 0.f};
    const f32 q = r / l;
    if (q >= 2.0f) return {0.f, 0.f, 0.f};
    const f32 sigma = 1.0f / (pi_f() * l * l * l);
    f32 dWdq;
    if (q < 1.0f) {
        dWdq = sigma * (-3.0f * q + 2.25f * q * q);
    } else {
        const f32 t = 2.0f - q;
        dWdq = sigma * (-0.75f) * t * t;
    }
    const f32 c = dWdq / (l * r);
    return {r_ij.x * c, r_ij.y * c, r_ij.z * c};
}

// Akinci 2013 cohesion kernel C(r, l) for surface tension.
__device__ inline f32 cohesion_C(f32 r, f32 l) {
    if (r <= 0.0f || r > l) return 0.0f;
    const f32 sigma = 32.0f / (pi_f() * powf(l, 9.0f));
    const f32 lm = l - r;
    const f32 t = lm * lm * lm * r * r * r;
    if (2.0f * r > l) return sigma * t;
    return sigma * (2.0f * t - powf(l, 6.0f) / 64.0f);
}

} // namespace water::sph

#pragma once

// Small POD types shared between host and device.
// Intentionally lightweight; we use float3/float4 from <cuda_runtime.h> on the
// device side and these here for code that doesn't pull in CUDA headers.

#include <cstdint>

namespace water {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

struct Vec3f { f32 x, y, z; };
struct Vec3i { i32 x, y, z; };
struct Vec4f { f32 x, y, z, w; };
struct Vec3u { u32 x, y, z; };

inline constexpr Vec3f kZero3f{0.0f, 0.0f, 0.0f};

} // namespace water

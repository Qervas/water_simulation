#pragma once

#include "water/core/types.h"
#include <algorithm>

namespace water {

class TimeStepper {
public:
    struct Config {
        f32 frame_dt    = 1.0f / 24.0f;  // 24 fps
        f32 dt_max      = 1.0f / 24.0f;
        f32 cfl_lambda  = 0.4f;
        u32 max_substeps = 16;
    };

    TimeStepper() = default;
    explicit TimeStepper(Config c) : cfg_(c) {}

    // Returns next substep dt given current max particle velocity (m/s) and
    // particle radius (m). The returned dt is clamped to keep substep counts
    // sane and to never exceed remaining-time-in-frame.
    f32 next_dt(f32 max_velocity, f32 particle_radius, f32 remaining_in_frame) const {
        f32 v = std::max(max_velocity, 1e-6f);
        f32 dt_cfl = cfg_.cfl_lambda * particle_radius / v;
        f32 dt_min = cfg_.frame_dt / static_cast<f32>(cfg_.max_substeps);
        return std::min({cfg_.dt_max, dt_cfl, remaining_in_frame, std::max(dt_cfl, dt_min)});
    }

    Config config() const noexcept { return cfg_; }

private:
    Config cfg_;
};

} // namespace water

#pragma once

#include "water/core/types.h"
#include <string>
#include <vector>

namespace water::scene {

struct Camera {
    Vec3f position{0.0f, 0.0f, 1.0f};
    Vec3f look_at{0.0f, 0.0f, 0.0f};
    Vec3f up{0.0f, 1.0f, 0.0f};
    f32   fov_y_deg = 30.0f;
};

struct OutputCfg {
    i32         width  = 1920;
    i32         height = 1080;
    f32         fps    = 24.0f;
    i32         frame_start = 0;
    i32         frame_end   = 120;
    std::string format = "exr";
    std::string video  = "mp4";
};

struct FluidCfg {
    std::string solver = "dfsph";
    f32 particle_radius   = 0.0015f;
    f32 rest_density      = 1000.0f;
    f32 viscosity         = 1e-3f;
    f32 surface_tension   = 0.0728f;
    Vec3f gravity{0.0f, -9.81f, 0.0f};
    Vec3f initial_block_min{0.4f, 0.05f, 0.4f};
    Vec3f initial_block_max{0.6f, 0.25f, 0.6f};
};

struct Scene {
    std::string name;
    OutputCfg   output;
    Camera      camera;
    FluidCfg    fluid;
};

// Throws std::runtime_error on parse error or missing required fields.
Scene load_scene(const std::string& path);

} // namespace water::scene

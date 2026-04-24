#include "water/scene/scene.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using nlohmann::json;

namespace water::scene {

namespace {

Vec3f read_vec3(const json& j, const char* key, Vec3f fallback) {
    if (!j.contains(key)) return fallback;
    auto a = j.at(key);
    return {a[0].get<f32>(), a[1].get<f32>(), a[2].get<f32>()};
}

void parse_camera(const json& j, Camera& cam) {
    if (!j.contains("camera")) return;
    const auto& c = j["camera"];
    cam.position    = read_vec3(c, "position", cam.position);
    cam.look_at     = read_vec3(c, "look_at",  cam.look_at);
    cam.up          = read_vec3(c, "up",       cam.up);
    if (c.contains("fov_y_deg")) cam.fov_y_deg = c["fov_y_deg"].get<f32>();
}

void parse_output(const json& j, OutputCfg& o) {
    if (!j.contains("output")) return;
    const auto& out = j["output"];
    if (out.contains("resolution")) {
        o.width  = out["resolution"][0].get<i32>();
        o.height = out["resolution"][1].get<i32>();
    }
    if (out.contains("fps"))         o.fps         = out["fps"].get<f32>();
    if (out.contains("frame_range")) {
        o.frame_start = out["frame_range"][0].get<i32>();
        o.frame_end   = out["frame_range"][1].get<i32>();
    }
    if (out.contains("format"))      o.format      = out["format"].get<std::string>();
    if (out.contains("video"))       o.video       = out["video"].get<std::string>();
}

void parse_fluid(const json& j, FluidCfg& f) {
    if (!j.contains("scene") || !j["scene"].contains("fluid")) return;
    const auto& fluid = j["scene"]["fluid"];
    if (fluid.contains("solver"))           f.solver           = fluid["solver"].get<std::string>();
    if (fluid.contains("particle_radius"))  f.particle_radius  = fluid["particle_radius"].get<f32>();
    if (fluid.contains("rest_density"))     f.rest_density     = fluid["rest_density"].get<f32>();
    if (fluid.contains("viscosity"))        f.viscosity        = fluid["viscosity"].get<f32>();
    if (fluid.contains("surface_tension"))  f.surface_tension  = fluid["surface_tension"].get<f32>();
    f.gravity            = read_vec3(fluid, "gravity",            f.gravity);
    f.initial_block_min  = read_vec3(fluid, "initial_block_min",  f.initial_block_min);
    f.initial_block_max  = read_vec3(fluid, "initial_block_max",  f.initial_block_max);
}

} // namespace

Scene load_scene(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("scene::load_scene: cannot open " + path);

    json j;
    try { in >> j; }
    catch (const json::parse_error& e) {
        throw std::runtime_error("scene::load_scene: JSON parse failed in " + path
                                  + ": " + e.what());
    }

    if (!j.contains("schema_version")) {
        throw std::runtime_error("scene::load_scene: missing schema_version in " + path);
    }
    auto sv = j["schema_version"].get<std::string>();
    if (sv != "1.0") {
        throw std::runtime_error("scene::load_scene: unsupported schema_version: " + sv);
    }

    Scene s;
    if (j.contains("name")) s.name = j["name"].get<std::string>();
    parse_output(j, s.output);
    parse_camera(j, s.camera);
    parse_fluid(j, s.fluid);
    return s;
}

} // namespace water::scene

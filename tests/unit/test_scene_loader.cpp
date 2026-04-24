#include <doctest/doctest.h>
#include "water/scene/scene.h"
#include <fstream>
#include <cstdio>

using namespace water::scene;

namespace {

void write_temp_file(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
}

} // namespace

TEST_CASE("scene::load_scene: minimal valid scene parses") {
    const std::string p = "/tmp/water_test_minimal.json";
    write_temp_file(p, R"({
        "schema_version": "1.0",
        "name": "test"
    })");
    auto s = load_scene(p);
    CHECK(s.name == "test");
    CHECK(s.output.width == 1920);          // default
    CHECK(s.fluid.solver == "dfsph");       // default
    std::remove(p.c_str());
}

TEST_CASE("scene::load_scene: missing schema_version throws") {
    const std::string p = "/tmp/water_test_no_schema.json";
    write_temp_file(p, R"({"name": "broken"})");
    CHECK_THROWS_AS(load_scene(p), std::runtime_error);
    std::remove(p.c_str());
}

TEST_CASE("scene::load_scene: unsupported schema version throws") {
    const std::string p = "/tmp/water_test_bad_schema.json";
    write_temp_file(p, R"({"schema_version": "99.0"})");
    CHECK_THROWS_AS(load_scene(p), std::runtime_error);
    std::remove(p.c_str());
}

TEST_CASE("scene::load_scene: file not found throws") {
    CHECK_THROWS_AS(load_scene("/nonexistent/path.json"), std::runtime_error);
}

TEST_CASE("scene::load_scene: full scene fields populate") {
    const std::string p = "/tmp/water_test_full.json";
    write_temp_file(p, R"({
        "schema_version": "1.0",
        "name": "full_test",
        "output": {
            "resolution": [800, 600],
            "fps": 30,
            "frame_range": [10, 20]
        },
        "camera": {
            "position": [1.0, 2.0, 3.0],
            "look_at": [0.0, 0.0, 0.0],
            "fov_y_deg": 45.0
        },
        "scene": {
            "fluid": {
                "particle_radius": 0.005,
                "gravity": [0.0, -9.81, 0.0]
            }
        }
    })");
    auto s = load_scene(p);
    CHECK(s.name == "full_test");
    CHECK(s.output.width == 800);
    CHECK(s.output.fps == doctest::Approx(30.0f));
    CHECK(s.camera.position.x == doctest::Approx(1.0f));
    CHECK(s.camera.fov_y_deg == doctest::Approx(45.0f));
    CHECK(s.fluid.particle_radius == doctest::Approx(0.005f));
    CHECK(s.fluid.gravity.y == doctest::Approx(-9.81f));
    std::remove(p.c_str());
}

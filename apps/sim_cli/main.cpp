#include "water/scene/scene.h"
#include "water/core/particle_store.h"
#include "water/core/timestep.h"
#include "water/core/cuda_check.h"
#include "water/solvers/dfsph.h"
#include "water/solvers/boundary_sampler.h"
#include <memory>
#include "water/renderer/vk_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

namespace {

struct Args {
    std::string scene_path;
    int         frames_start = -1;        // -1 = use scene
    int         frames_end   = -1;
    bool        record       = false;
    bool        skip_vulkan  = false;     // skip Vulkan device creation (faster startup)
    std::string out_dir      = "out";
    float       damping      = -1.0f;     // <0 = solver default
    float       viscosity    = -1.0f;     // <0 = use scene value
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--scene"  && i + 1 < argc) { a.scene_path = argv[++i]; }
        else if (s == "--frames" && i + 1 < argc) {
            std::string r = argv[++i];
            auto colon = r.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "--frames expects START:END\n"); std::exit(2);
            }
            a.frames_start = std::stoi(r.substr(0, colon));
            a.frames_end   = std::stoi(r.substr(colon + 1));
        }
        else if (s == "--record")             { a.record = true; }
        else if (s == "--out" && i + 1 < argc){ a.out_dir = argv[++i]; }
        else if (s == "--no-vulkan")          { a.skip_vulkan = true; }
        else if (s == "--damping" && i + 1 < argc)   { a.damping   = std::stof(argv[++i]); }
        else if (s == "--viscosity" && i + 1 < argc) { a.viscosity = std::stof(argv[++i]); }
        else if (s == "-h" || s == "--help") {
            std::puts("Usage: sim_cli --scene PATH [--frames START:END] [--record] "
                      "[--out DIR] [--no-vulkan] [--damping F] [--viscosity F]");
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", s.c_str()); std::exit(2);
        }
    }
    if (a.scene_path.empty()) {
        std::fprintf(stderr, "Error: --scene required\n"); std::exit(2);
    }
    return a;
}

std::vector<water::Vec3f> generate_initial_block(const water::scene::FluidCfg& f) {
    std::vector<water::Vec3f> pts;
    const float spacing = 2.0f * f.particle_radius;
    for (float x = f.initial_block_min.x; x <= f.initial_block_max.x; x += spacing)
    for (float y = f.initial_block_min.y; y <= f.initial_block_max.y; y += spacing)
    for (float z = f.initial_block_min.z; z <= f.initial_block_max.z; z += spacing) {
        pts.push_back({x, y, z});
    }
    return pts;
}

void write_frame_binary(const std::string& path, const water::Vec3f* positions,
                         std::size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open " + path + " for writing");
    std::uint32_t n = static_cast<std::uint32_t>(count);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(positions), sizeof(water::Vec3f) * count);
}

} // namespace

int main(int argc, char** argv) try {
    auto args  = parse(argc, argv);
    auto scene = water::scene::load_scene(args.scene_path);

    if (args.frames_start < 0) args.frames_start = scene.output.frame_start;
    if (args.frames_end   < 0) args.frames_end   = scene.output.frame_end;

    std::printf("=== water_sim sim_cli ===\n");
    std::printf("scene:        %s\n", scene.name.c_str());
    std::printf("frames:       %d..%d (%d total)\n",
                args.frames_start, args.frames_end,
                args.frames_end - args.frames_start);
    std::printf("solver:       %s\n", scene.fluid.solver.c_str());
    std::printf("record:       %s%s\n",
                args.record ? "yes -> " : "no",
                args.record ? args.out_dir.c_str() : "");

    if (!args.skip_vulkan) {
        water::renderer::VulkanDevice vk;
        auto info = vk.info();
        std::printf("vulkan:       %s (api %u.%u.%u, RT %s)\n",
                    info.device_name.c_str(),
                    VK_API_VERSION_MAJOR(info.api_version),
                    VK_API_VERSION_MINOR(info.api_version),
                    VK_API_VERSION_PATCH(info.api_version),
                    info.ray_tracing_supported ? "yes" : "no");
    }

    auto initial = generate_initial_block(scene.fluid);
    if (initial.empty()) { std::fprintf(stderr, "no particles\n"); return 1; }
    std::printf("particles:    %zu\n", initial.size());

    water::ParticleStore store(initial.size());
    store.resize(initial.size());
    WATER_CUDA_CHECK(cudaMemcpy(store.positions(), initial.data(),
                                 sizeof(water::Vec3f) * initial.size(),
                                 cudaMemcpyHostToDevice));

    water::solvers::DFSPHSolver::Config cfg;
    cfg.rest_density     = scene.fluid.rest_density;
    cfg.particle_radius  = scene.fluid.particle_radius;
    cfg.smoothing_length = 2.0f * scene.fluid.particle_radius;
    cfg.viscosity        = (args.viscosity > 0.0f)
                            ? args.viscosity
                            : std::max(scene.fluid.viscosity, 1e-2f);
    cfg.surface_tension  = scene.fluid.surface_tension;
    cfg.gravity          = scene.fluid.gravity;
    cfg.domain_min       = {0.0f, 0.0f, 0.0f};
    cfg.domain_max       = {1.0f, 1.0f, 1.0f};
    if (args.damping >= 0.0f) cfg.damping = args.damping;
    std::printf("solver cfg:   damping=%.3f viscosity=%.4f\n",
                cfg.damping, cfg.viscosity);

    // Boundary particles for the AABB box.
    const float spacing = 2.0f * cfg.particle_radius;
    // 3 layers required because kernel support (2*l = 2*spacing) needs
    // ceil(2*l/spacing) layers to fully sample the wall side of a fluid
    // particle right at the boundary.
    auto bpts = water::solvers::sample_aabb_boundary(
        cfg.domain_min, cfg.domain_max, spacing, 3);
    std::printf("boundary:     %zu particles (Akinci 2012, 2 layers)\n", bpts.size());
    auto boundary = std::make_unique<water::ParticleStore>(bpts.size());
    boundary->resize(bpts.size());
    WATER_CUDA_CHECK(cudaMemcpy(boundary->positions(), bpts.data(),
                                 sizeof(water::Vec3f) * bpts.size(),
                                 cudaMemcpyHostToDevice));

    water::solvers::DFSPHSolver solver(store, boundary.get(), cfg);
    water::TimeStepper ts;

    if (args.record) {
        // Best-effort mkdir -p; ignore EEXIST.
        std::string cmd = "mkdir -p " + args.out_dir;
        std::system(cmd.c_str());
        // Dump frame 0 (initial state) before stepping.
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/frame_%04d.bin",
                      args.out_dir.c_str(), args.frames_start);
        write_frame_binary(fname, initial.data(), initial.size());
    }

    const float frame_dt = 1.0f / scene.output.fps;
    std::vector<water::Vec3f> pos_host(store.count());

    for (int f = args.frames_start; f < args.frames_end; ++f) {
        float t_remaining = frame_dt;
        int sub = 0;
        while (t_remaining > 1e-6f && sub < 64) {
            // Use a fixed timestep early (max_velocity is 0 before first step).
            float dt = ts.next_dt(/*max_v=*/std::max(solver.max_velocity(), 0.5f),
                                  cfg.particle_radius, t_remaining);
            solver.step(dt);
            t_remaining -= dt;
            ++sub;
        }
        WATER_CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("frame %4d: substeps=%2d, max_v=%.4f m/s\n",
                    f + 1, sub, solver.max_velocity());

        if (args.record) {
            WATER_CUDA_CHECK(cudaMemcpy(pos_host.data(), store.positions(),
                                         sizeof(water::Vec3f) * store.count(),
                                         cudaMemcpyDeviceToHost));
            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/frame_%04d.bin",
                          args.out_dir.c_str(), f + 1);
            write_frame_binary(fname, pos_host.data(), pos_host.size());
        }
    }

    std::printf("=== sim_cli OK ===\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "sim_cli error: %s\n", e.what());
    return 1;
}

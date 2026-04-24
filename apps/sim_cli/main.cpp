#include "water/scene/scene.h"
#include "water/core/particle_store.h"
#include "water/core/spatial_accel.h"
#include "water/core/timestep.h"
#include "water/core/cuda_check.h"
#include "water/renderer/vk_device.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

struct Args {
    std::string scene_path;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--scene" && i + 1 < argc) { a.scene_path = argv[++i]; }
        else if (s == "-h" || s == "--help") {
            std::puts("Usage: sim_cli --scene PATH");
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", s.c_str());
            std::exit(2);
        }
    }
    if (a.scene_path.empty()) {
        std::fprintf(stderr, "Error: --scene required\n");
        std::exit(2);
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

} // namespace

int main(int argc, char** argv) {
    try {
        auto args  = parse(argc, argv);
        auto scene = water::scene::load_scene(args.scene_path);

        std::printf("=== water_sim sim_cli ===\n");
        std::printf("scene:        %s\n", scene.name.c_str());
        std::printf("output:       %dx%d @ %.1f fps, frames %d..%d\n",
                    scene.output.width, scene.output.height,
                    scene.output.fps,
                    scene.output.frame_start, scene.output.frame_end);
        std::printf("solver:       %s\n", scene.fluid.solver.c_str());
        std::printf("particle r:   %.4f m\n", scene.fluid.particle_radius);
        std::printf("rest density: %.1f kg/m^3\n", scene.fluid.rest_density);

        // Vulkan check
        water::renderer::VulkanDevice vk;
        auto info = vk.info();
        std::printf("vulkan:       %s (api %u.%u.%u, RT %s)\n",
                    info.device_name.c_str(),
                    VK_API_VERSION_MAJOR(info.api_version),
                    VK_API_VERSION_MINOR(info.api_version),
                    VK_API_VERSION_PATCH(info.api_version),
                    info.ray_tracing_supported ? "yes" : "no");

        // Initial particle block
        auto initial = generate_initial_block(scene.fluid);
        std::printf("particles:    %zu (initial block)\n", initial.size());

        if (initial.empty()) {
            std::fprintf(stderr, "warning: zero particles generated\n");
            return 0;
        }

        water::ParticleStore store(initial.size() * 2);  // 2x headroom
        store.resize(initial.size());
        WATER_CUDA_CHECK(cudaMemcpy(store.positions(), initial.data(),
                                     sizeof(water::Vec3f) * initial.size(),
                                     cudaMemcpyHostToDevice));

        // Build LBVH on the initial particles.
        water::AABB scene_bounds{
            {scene.fluid.initial_block_min.x - 0.1f,
             scene.fluid.initial_block_min.y - 0.1f,
             scene.fluid.initial_block_min.z - 0.1f},
            {scene.fluid.initial_block_max.x + 0.1f,
             scene.fluid.initial_block_max.y + 0.1f,
             scene.fluid.initial_block_max.z + 0.1f}};
        water::SpatialAccel accel;
        accel.build(store.positions(), store.count(), scene_bounds);
        WATER_CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("lbvh leaves:  %zu\n", accel.leaf_count());

        // Substepping smoke check
        water::TimeStepper ts;
        const auto dt = ts.next_dt(/*max_v=*/1.0f, scene.fluid.particle_radius, 1.0f / 24.0f);
        std::printf("first dt:     %.6f s (CFL with v=1 m/s)\n", dt);

        std::printf("=== sim_cli OK ===\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sim_cli error: %s\n", e.what());
        return 1;
    }
}

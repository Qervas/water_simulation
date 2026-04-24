# core/CMakeLists.txt

# CUDAToolkit's IMPORTED targets are scoped to the directory they're found
# in. third_party/ called find_package but the CUDA::cudart target isn't
# inherited here — re-run find_package locally. (Idempotent.)
find_package(CUDAToolkit REQUIRED)

# water_core: solver-agnostic CUDA + C++ infrastructure shared by all
# downstream modules (solvers, surface, scene, renderer, viewport).
add_library(water_core STATIC
    src/particle_store.cu
    src/spatial_accel.cu
    src/cell_grid.cu
    src/boundary.cu
    src/timestep.cpp
)

target_include_directories(water_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(water_core
    PUBLIC
        glm::glm
        CUDA::cudart
)

# CUDA-specific properties
set_target_properties(water_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE   ON
)

# Allow downstream targets to use lambda in __device__ contexts
target_compile_options(water_core PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

include(FetchContent)

# Suppress noise from third-party CMake
set(FETCHCONTENT_QUIET FALSE)

# ---- glm: header-only math (CPU side only; do NOT use in CUDA kernels) ----
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glm)

# ---- nlohmann/json: scene file parsing ----
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ---- doctest: testing framework ----
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE
)
set(DOCTEST_WITH_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(doctest)

# ---- stb: image loading (single-header) ----
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# ---- tinyobjloader: OBJ mesh load ----
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tinyobjloader)

# ---- Vulkan: system SDK ----
find_package(Vulkan REQUIRED)
message(STATUS "  Vulkan SDK:        ${Vulkan_VERSION} (${Vulkan_INCLUDE_DIR})")

# ---- Slang: prefer system slangc on PATH ----
find_program(SLANGC_EXECUTABLE slangc REQUIRED)
message(STATUS "  Slang compiler:    ${SLANGC_EXECUTABLE}")

# ---- CUDA Toolkit (in addition to language enable) ----
find_package(CUDAToolkit REQUIRED)
message(STATUS "  CUDA Toolkit:      ${CUDAToolkit_VERSION} (${CUDAToolkit_INCLUDE_DIRS})")

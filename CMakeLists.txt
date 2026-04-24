cmake_minimum_required(VERSION 3.27)

# Some FetchContent dependencies have very old cmake_minimum_required values
# (doctest, glm, etc.) that CMake 4+ refuses to honor. Setting a minimum
# policy version makes them configurable again.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "")

# CUDA 13.2 supports gcc up to 15. If your system gcc is newer (e.g. Fedora 44
# ships gcc 16), set CMAKE_CUDA_HOST_COMPILER to point at a supported gcc.
# See CMakeUserPresets.json (gitignored) for an example using a conda-forge
# gcc 14 toolchain.

project(water_sim
    VERSION 2.0.0
    DESCRIPTION "Modern GPU-native water simulation and offline cinematic renderer"
    LANGUAGES CXX CUDA
)

# ---- Standards ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Blackwell (RTX 50-series). Override with -DCMAKE_CUDA_ARCHITECTURES=89 for Ada, etc.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Default to Release if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Build options ----
option(WATER_BUILD_TESTS    "Build unit and regression tests" ON)
option(WATER_BUILD_VIEWPORT "Build interactive dev viewport"  ON)
option(WATER_BUILD_SIM_CLI  "Build offline simulation CLI"    ON)
option(WATER_ENABLE_ASAN    "Enable AddressSanitizer (debug)" OFF)

# ---- Compile flags ----
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    )
endif()

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -use_fast_math")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -G enables full device debug. Cannot combine with -lineinfo.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# Force colored diagnostics under Ninja
if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
endif()

# ---- Output dirs ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---- Subdirectories ----
add_subdirectory(third_party)
add_subdirectory(core)
add_subdirectory(scene)
add_subdirectory(solvers)
add_subdirectory(renderer)
add_subdirectory(apps)

if(WATER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Status ----
message(STATUS "")
message(STATUS "==== water_sim configuration ====")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests:             ${WATER_BUILD_TESTS}")
message(STATUS "  Viewport:          ${WATER_BUILD_VIEWPORT}")
message(STATUS "  sim_cli:           ${WATER_BUILD_SIM_CLI}")
message(STATUS "==================================")
message(STATUS "")

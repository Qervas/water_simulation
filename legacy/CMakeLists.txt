## Clean CMake configuration (Windows focus). No CMake CUDA language enable.
cmake_minimum_required(VERSION 3.18)
project(water_simulation LANGUAGES CXX)

# ---- Build type & C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ---- Ensure MSVC on Windows ----
if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR "Windows build requires MSVC (generate with Visual Studio or run from Developer Command Prompt).")
endif()

find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "nvcc not found in PATH. Ensure CUDA_PATH/bin is on PATH.")
endif()
get_filename_component(CUDA_BIN_DIR "${NVCC}" DIRECTORY)
get_filename_component(CUDA_ROOT "${CUDA_BIN_DIR}/.." ABSOLUTE)
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
set(CUDA_LIB_DIR "${CUDA_ROOT}/lib/x64")
if(NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda_runtime.h")
    message(FATAL_ERROR "Invalid CUDA include directory: ${CUDA_INCLUDE_DIR}")
endif()

# ---- NVCC flags ----
set(CUDA_ARCH "89" CACHE STRING "Single SM architecture code (e.g. 61,75,89)")
set(NVCC_BASE_FLAGS --expt-extended-lambda -use_fast_math -arch=sm_${CUDA_ARCH} -std=c++20 -Xcompiler /std:c++20)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND NVCC_BASE_FLAGS -G)
endif()

# ---- Source collection ----
include(FetchContent)
# GLEW (via CMake-friendly fork)
if(WIN32)
    # Build static glew via Perlmint/glew-cmake to avoid prebuilt binaries
    FetchContent_Declare(
        glew
        GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(glew)
endif()

file(GLOB CPP_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB CU_SOURCES  CONFIGURE_DEPENDS src/*.cu)

# ---- Header-only dependencies (FetchContent) ----
include(FetchContent)
set(GLM_VERSION_TAG "1.0.1" CACHE STRING "GLM git tag/branch")
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        ${GLM_VERSION_TAG}
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(glm)
set(GLM_INCLUDE_DIR ${glm_SOURCE_DIR})

# stb (image loading)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)
set(STB_INCLUDE_DIR ${stb_SOURCE_DIR})

# ---- Custom compilation of CUDA sources to objects ----
set(CUDA_OBJECTS)
foreach(cu ${CU_SOURCES})
    get_filename_component(fname ${cu} NAME_WE)
    set(obj "${CMAKE_CURRENT_BINARY_DIR}/cuda/${fname}.obj")
    add_custom_command(
        OUTPUT ${obj}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/cuda"
        COMMAND ${NVCC} ${NVCC_BASE_FLAGS} -I"${CUDA_INCLUDE_DIR}" -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_SOURCE_DIR}/3rd-party" -I"${GLM_INCLUDE_DIR}" -I"${STB_INCLUDE_DIR}" -x cu -c ${cu} -o ${obj} -Xcompiler /MD$<$<CONFIG:Debug>:d>
        DEPENDS ${cu}
        COMMENT "NVCC compile ${fname}.cu -> ${fname}.obj"
        VERBATIM
    )
    list(APPEND CUDA_OBJECTS ${obj})
endforeach()

add_custom_target(cuda_objs DEPENDS ${CUDA_OBJECTS})

# ---- Executable ----
add_executable(app ${CPP_SOURCES} ${CUDA_OBJECTS})
add_dependencies(app cuda_objs)

# ---- Includes ----
target_include_directories(app PRIVATE include 3rd-party ${CUDA_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${STB_INCLUDE_DIR})

# ---- Libraries (Windows only subset) ----
if(WIN32)
    target_link_directories(app PRIVATE ${CUDA_LIB_DIR} ${CMAKE_SOURCE_DIR}/3rd-party/lib/x64)
    # GLFW (MSVC) from Scoop
    if(NOT DEFINED GLFW_ROOT AND DEFINED ENV{USERPROFILE})
        set(GLFW_ROOT "$ENV{USERPROFILE}/scoop/apps/glfw/current")
    endif()
    if(EXISTS "${GLFW_ROOT}/include/GLFW/glfw3.h")
        target_include_directories(app PRIVATE ${GLFW_ROOT}/include)
        # Prefer VS 2022 libs if available
        if(EXISTS "${GLFW_ROOT}/lib-vc2022/glfw3.lib")
            target_link_directories(app PRIVATE ${GLFW_ROOT}/lib-vc2022)
            find_library(GLFW3_LIB glfw3 PATHS ${GLFW_ROOT}/lib-vc2022 NO_DEFAULT_PATH)
        else()
            find_library(GLFW3_LIB glfw3 PATHS ${GLFW_ROOT} NO_DEFAULT_PATH)
        endif()
        if(GLFW3_LIB)
            target_link_libraries(app PRIVATE ${GLFW3_LIB})
        else()
            message(WARNING "glfw3.lib not found in ${GLFW_ROOT}")
        endif()
    else()
        message(WARNING "GLFW not found at ${GLFW_ROOT}; install via 'scoop install glfw' if needed.")
    endif()
    # GLEW from source (libglew_static/libglew_shared targets)
    if(TARGET libglew_static)
        target_link_libraries(app PRIVATE libglew_static)
        target_compile_definitions(app PRIVATE GLEW_STATIC)
    elseif(TARGET libglew_shared)
        target_link_libraries(app PRIVATE libglew_shared)
    else()
        message(WARNING "GLEW source targets not found; ensure FetchContent for glew executed earlier.")
    endif()
    find_library(FREEGLUT_LIB freeglut PATHS ${CMAKE_SOURCE_DIR}/3rd-party/lib/x64)
    if(FREEGLUT_LIB)
        target_link_libraries(app PRIVATE ${FREEGLUT_LIB})
    endif()
    # System OpenGL
    target_link_libraries(app PRIVATE opengl32)
    # CUDA runtime
    find_library(CUDART_LIB cudart PATHS ${CUDA_LIB_DIR})
    if(CUDART_LIB)
        target_link_libraries(app PRIVATE ${CUDART_LIB})
    else()
        message(WARNING "cudart.lib not found in ${CUDA_LIB_DIR}")
    endif()
endif()

# ---- Warnings & definitions ----
if(MSVC)
    target_compile_options(app PRIVATE /W4)
else()
    target_compile_options(app PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_compile_definitions(app PRIVATE $<$<CONFIG:Debug>:DEBUG> GLEW_STATIC)

# ---- Resource directories copy (portable) ----
foreach(dir shaders texture)
    if(EXISTS ${CMAKE_SOURCE_DIR}/${dir})
        add_custom_command(TARGET app PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/${dir} ${CMAKE_BINARY_DIR}/${dir})
    endif()
endforeach()

# ---- Status messages ----
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "NVCC: ${NVCC}")
message(STATUS "CUDA root: ${CUDA_ROOT}")
message(STATUS "CUDA arch: ${CUDA_ARCH}")
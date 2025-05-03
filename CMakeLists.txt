## NOTICE: This CMakeLists.txt is for Linux only

cmake_minimum_required(VERSION 3.18)

# We need to set these before the project call
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")

# Set CUDA architectures before project()
# 89 = RTX 4060 Mobile GPU (Ada Lovelace architecture)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Project declaration
project(app LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
                     -G \
                     -gencode arch=compute_89,code=sm_89 \
                     --expt-extended-lambda \
                     -use_fast_math \
                     -allow-unsupported-compiler \
                     -Xcompiler -Wall,-Wextra \
                     -diag-suppress 20012")

# Language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    /usr/include 
    /usr/local/cuda/include
    3rd-party 
    include
)

# Library directories
link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/cuda/lib64
)

# Build type
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Source files
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# Executable
add_executable(app ${SOURCES} src/main.cpp)

# Libraries
target_link_libraries(app -lstdc++fs glfw GLEW GL GLU glut -lcudart)

# Include directories for target
target_include_directories(app PUBLIC include)

# Compiler options for target
target_compile_options(app PUBLIC -Wall -Wextra)

# Compiler definitions for target
target_compile_definitions(app PUBLIC -DDEBUG)

# Symlinks for content directories
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders SYMBOLIC)
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/texture ${CMAKE_BINARY_DIR}/texture SYMBOLIC)
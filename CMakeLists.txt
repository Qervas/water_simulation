## NOTICE: This CMakeLists.txt is for Linux only

cmake_minimum_required(VERSION 3.0)


set(GCC_PATH "ENTER YOUR GCC PATH HERE")
set(CMAKE_C_COMPILER "${GCC_PATH}/bin/gcc")
set(CMAKE_CXX_COMPILER "${GCC_PATH}/bin/g++")

# # Set C and C++ compilers for CUDA
set(CMAKE_CUDA_HOST_COMPILER "${GCC_PATH}/bin/g++" CACHE STRING "")

set(CUDA_HOME "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CUDA_ARCHITECTURES 60)

project(app CXX CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -gencode arch=compute_60,code=sm_60 \
					--expt-extended-lambda -use_fast_math -Xcompiler -Wall,-Wextra \
					-diag-suppress 20012 \
					")
# -Wno-unused-parameter,-Wno-unused-function,-Wno-unused-variable \


set(CMAKE_CXX_STANDARD 20)

# include files in "include" folder

include_directories( ${CUDA_HOME}/include 3rd-party include )
link_directories(${CUDA_HOME}/lib64 )

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# add all source files in "src" folder
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# add executable
add_executable(app ${SOURCES} src/main.cpp)

# add libraries
target_link_libraries(app -lstdc++fs glfw GLEW GL GLU glut cudart )

# add include files
target_include_directories(app PUBLIC include)

# add compiler flags
target_compile_options(app PUBLIC -Wall -Wextra )

# add compiler definitions
target_compile_definitions(app PUBLIC -DDEBUG)

#symlink  content under ./shaders to build, not the folder itself
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders SYMBOLIC)
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/texture ${CMAKE_BINARY_DIR}/texture SYMBOLIC)
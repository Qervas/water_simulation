cmake_minimum_required(VERSION 3.0)

# set(GCC_12 "/home/franktudor/Application/Library/gcc-12.2.0/build/gcc-12.2.0")
# set(CMAKE_C_COMPILER "${GCC_12}/bin/gcc")
# set(CMAKE_CXX_COMPILER "${GCC_12}/bin/g++")

# Set C and C++ compilers for CUDA
# set(CMAKE_CUDA_HOST_COMPILER "${GCC_12}/bin/gcc" CACHE STRING "")

set(CUDA_HOME "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")
set(CUDAToolkit_ROOT ${CUDA_HOME})
set(CMAKE_CUDA_ARCHITECTURES 60)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 \
					--expt-extended-lambda -use_fast_math -Xcompiler -Wall,-Wextra, \
					-fpermissive \
					")
					# -Wno-unused-parameter,-Wno-unused-function,-Wno-unused-variable \

project(app CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# include files in "include" folder
include_directories(include)
include_directories(${CUDA_HOME}/include /usr/local/cuda-samples/Common)
link_directories(${CUDA_HOME}/lib64 /usr/local/cuda-samples/Common/lib/x64/)



# add all source files in "src" folder
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# add executable
add_executable(app ${SOURCES})

# add libraries
target_link_libraries(app -lstdc++fs glfw GLEW GL GLU glut cudart)

# add include files
target_include_directories(app PUBLIC include)

# add compiler flags
target_compile_options(app PUBLIC -Wall -Wextra )
 
# add compiler definitions
target_compile_definitions(app PUBLIC -DDEBUG)

#symlink  content under ./shaders to build, not the folder itself
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders SYMBOLIC)
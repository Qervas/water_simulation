add_library(water_dfsph STATIC
    src/dfsph.cu
    src/dfsph_kernels.cu
    src/boundary_sampler.cpp
)

target_include_directories(water_dfsph
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(water_dfsph
    PUBLIC water_core
)

set_target_properties(water_dfsph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE   ON
)

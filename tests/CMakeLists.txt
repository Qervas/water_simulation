# tests/CMakeLists.txt
add_executable(water_tests
    unit/test_main.cpp
    unit/test_particle_store.cu
    unit/test_spatial_accel.cu
    unit/test_boundary.cu
    unit/test_scene_loader.cpp
)

target_link_libraries(water_tests
    PRIVATE
        water_core
        water_scene
        doctest::doctest
)

set_target_properties(water_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME water_tests COMMAND water_tests)

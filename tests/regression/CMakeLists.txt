# tests/regression/CMakeLists.txt
add_executable(water_regression
    test_settled_puddle.cu
    ../unit/test_main.cpp
)

target_link_libraries(water_regression PRIVATE
    water_core
    water_scene
    water_dfsph
    doctest::doctest
)

set_target_properties(water_regression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME water_regression COMMAND water_regression)

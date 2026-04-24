# apps/CMakeLists.txt
if(WATER_BUILD_SIM_CLI)
    add_executable(sim_cli sim_cli/main.cpp)
    target_link_libraries(sim_cli PRIVATE
        water_core
        water_scene
        water_dfsph
        water_renderer
    )
endif()

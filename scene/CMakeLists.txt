# scene/CMakeLists.txt
add_library(water_scene STATIC
    src/loader.cpp
)

target_include_directories(water_scene
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(water_scene
    PUBLIC
        water_core
        nlohmann_json::nlohmann_json
)

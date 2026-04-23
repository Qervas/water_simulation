# renderer/CMakeLists.txt

# ---- Slang shader compilation ----
# For each .slang file, invoke `slangc` at build time to produce a .spv blob.
file(GLOB_RECURSE SLANG_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang)

set(SLANG_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SLANG_OUTPUT_DIR})

set(SLANG_OUTPUTS)
foreach(slang_src ${SLANG_SOURCES})
    get_filename_component(name ${slang_src} NAME_WE)
    set(spv_out ${SLANG_OUTPUT_DIR}/${name}.spv)
    add_custom_command(
        OUTPUT  ${spv_out}
        COMMAND ${SLANGC_EXECUTABLE}
                -profile sm_6_5
                -target spirv
                -emit-spirv-directly
                -entry main
                -o ${spv_out}
                ${slang_src}
        DEPENDS ${slang_src}
        COMMENT "Slang -> SPIR-V: ${name}.slang"
        VERBATIM
    )
    list(APPEND SLANG_OUTPUTS ${spv_out})
endforeach()

add_custom_target(water_shaders DEPENDS ${SLANG_OUTPUTS})

# ---- Vulkan + renderer library ----
find_package(Vulkan REQUIRED)

add_library(water_renderer STATIC
    src/vk_device.cpp
)

target_include_directories(water_renderer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(water_renderer
    PUBLIC
        water_core
        Vulkan::Vulkan
)

add_dependencies(water_renderer water_shaders)

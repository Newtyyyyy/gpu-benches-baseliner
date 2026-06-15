cmake_minimum_required(VERSION 3.15)

add_library(gpu-umstream-hip-obj OBJECT GpuUmstreamWorkload.hip)

target_include_directories(gpu-umstream-hip-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-umstream-hip-obj)
target_link_libraries(gpu-umstream-hip-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-umstream-hip $<TARGET_OBJECTS:gpu-umstream-hip-obj>)
    target_link_libraries(gpu-umstream-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-umstream-hip PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

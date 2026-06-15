cmake_minimum_required(VERSION 3.15)

add_library(gpu-memcpy-hip-obj OBJECT GpuMemcpyWorkload.hip)

target_include_directories(gpu-memcpy-hip-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-memcpy-hip-obj)
target_link_libraries(gpu-memcpy-hip-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-memcpy-hip $<TARGET_OBJECTS:gpu-memcpy-hip-obj>)
    target_link_libraries(gpu-memcpy-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-memcpy-hip PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

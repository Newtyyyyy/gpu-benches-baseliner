cmake_minimum_required(VERSION 3.15)

add_library(gpu-cache-hip-obj OBJECT GpuCacheWorkload.hip)

target_include_directories(gpu-cache-hip-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-cache-hip-obj)
target_link_libraries(gpu-cache-hip-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-cache-hip $<TARGET_OBJECTS:gpu-cache-hip-obj>)
    target_link_libraries(gpu-cache-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-cache-hip PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()
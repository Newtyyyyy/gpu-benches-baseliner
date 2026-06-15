cmake_minimum_required(VERSION 3.15)

add_library(gpu-incore-hip-obj OBJECT GpuIncoreWorkload.hip)

target_include_directories(gpu-incore-hip-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-incore-hip-obj)
target_link_libraries(gpu-incore-hip-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-incore-hip $<TARGET_OBJECTS:gpu-incore-hip-obj>)
    target_link_libraries(gpu-incore-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-incore-hip PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()
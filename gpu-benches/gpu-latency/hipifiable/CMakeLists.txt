cmake_minimum_required(VERSION 3.15)

add_library(gpu-latency-hipifiable-obj OBJECT GpuLatencyWorkload.hip)

target_include_directories(gpu-latency-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-latency-hipifiable-obj)
target_link_libraries(gpu-latency-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-latency-hipifiable $<TARGET_OBJECTS:gpu-latency-hipifiable-obj>)
    target_link_libraries(gpu-latency-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-latency-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

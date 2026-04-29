cmake_minimum_required(VERSION 3.15)

add_library(gpu-latency-hip-obj OBJECT gpu-latency.hip)
target_include_directories(gpu-latency-hip-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-latency-hip-obj)
target_link_libraries(gpu-latency-hip-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-latency-hip $<TARGET_OBJECTS:gpu-latency-hip-obj>)
    target_link_libraries(gpu-latency-hip PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-latency-hip PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

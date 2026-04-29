cmake_minimum_required(VERSION 3.15)

add_library(gpu-latency-cuda-obj OBJECT gpu-latency.cu)

target_include_directories(gpu-latency-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-latency-cuda-obj)
target_link_libraries(gpu-latency-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-latency-cuda $<TARGET_OBJECTS:gpu-latency-cuda-obj>)
    target_link_libraries(gpu-latency-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-latency-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
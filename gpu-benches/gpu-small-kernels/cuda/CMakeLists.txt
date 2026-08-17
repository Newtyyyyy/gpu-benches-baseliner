cmake_minimum_required(VERSION 3.15)

add_library(gpu-small-kernels-cuda-obj OBJECT GpuSmallKernelsWorkload.cu)

target_include_directories(gpu-small-kernels-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-small-kernels-cuda-obj)
target_link_libraries(gpu-small-kernels-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-small-kernels-cuda $<TARGET_OBJECTS:gpu-small-kernels-cuda-obj>)
    target_link_libraries(gpu-small-kernels-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-small-kernels-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

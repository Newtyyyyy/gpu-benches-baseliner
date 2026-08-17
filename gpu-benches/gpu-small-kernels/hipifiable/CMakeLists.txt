cmake_minimum_required(VERSION 3.15)

add_library(gpu-small-kernels-hipifiable-obj OBJECT GpuSmallKernelsWorkload.hip)

target_include_directories(gpu-small-kernels-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-small-kernels-hipifiable-obj)
target_link_libraries(gpu-small-kernels-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-small-kernels-hipifiable $<TARGET_OBJECTS:gpu-small-kernels-hipifiable-obj>)
    target_link_libraries(gpu-small-kernels-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-small-kernels-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

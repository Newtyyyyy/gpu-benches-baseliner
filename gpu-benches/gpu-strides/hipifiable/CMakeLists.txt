cmake_minimum_required(VERSION 3.15)

add_library(gpu-strides-hipifiable-obj OBJECT GpuStridesWorkload.hip)

target_include_directories(gpu-strides-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-strides-hipifiable-obj)
target_link_libraries(gpu-strides-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-strides-hipifiable $<TARGET_OBJECTS:gpu-strides-hipifiable-obj>)
    target_link_libraries(gpu-strides-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-strides-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

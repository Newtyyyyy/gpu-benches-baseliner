cmake_minimum_required(VERSION 3.15)

add_library(gpu-incore-hipifiable-obj OBJECT GpuIncoreWorkload.hip)

target_include_directories(gpu-incore-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-incore-hipifiable-obj)
target_link_libraries(gpu-incore-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-incore-hipifiable $<TARGET_OBJECTS:gpu-incore-hipifiable-obj>)
    target_link_libraries(gpu-incore-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-incore-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

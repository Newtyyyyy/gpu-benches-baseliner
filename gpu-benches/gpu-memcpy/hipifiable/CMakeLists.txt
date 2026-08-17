cmake_minimum_required(VERSION 3.15)

add_library(gpu-memcpy-hipifiable-obj OBJECT GpuMemcpyWorkload.hip)

target_include_directories(gpu-memcpy-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-memcpy-hipifiable-obj)
target_link_libraries(gpu-memcpy-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-memcpy-hipifiable $<TARGET_OBJECTS:gpu-memcpy-hipifiable-obj>)
    target_link_libraries(gpu-memcpy-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-memcpy-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

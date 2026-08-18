cmake_minimum_required(VERSION 3.15)

add_library(gpu-umstream-hipifiable-obj OBJECT GpuUmstreamWorkload.hip)

target_include_directories(gpu-umstream-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-umstream-hipifiable-obj)
target_link_libraries(gpu-umstream-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-umstream-hipifiable $<TARGET_OBJECTS:gpu-umstream-hipifiable-obj>)
    target_link_libraries(gpu-umstream-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-umstream-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

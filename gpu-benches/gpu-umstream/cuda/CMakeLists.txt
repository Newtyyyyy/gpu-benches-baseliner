cmake_minimum_required(VERSION 3.15)

add_library(gpu-umstream-cuda-obj OBJECT GpuUmstreamWorkload.cu)

target_include_directories(gpu-umstream-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-umstream-cuda-obj)
target_link_libraries(gpu-umstream-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-umstream-cuda $<TARGET_OBJECTS:gpu-umstream-cuda-obj>)
    target_link_libraries(gpu-umstream-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-umstream-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()
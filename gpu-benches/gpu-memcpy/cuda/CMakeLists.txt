cmake_minimum_required(VERSION 3.15)

add_library(gpu-memcpy-cuda-obj OBJECT GpuMemcpyWorkload.cu)

target_include_directories(gpu-memcpy-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-memcpy-cuda-obj)
target_link_libraries(gpu-memcpy-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-memcpy-cuda $<TARGET_OBJECTS:gpu-memcpy-cuda-obj>)
    target_link_libraries(gpu-memcpy-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-memcpy-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

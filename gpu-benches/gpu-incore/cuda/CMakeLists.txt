cmake_minimum_required(VERSION 3.15)

add_library(gpu-incore-cuda-obj OBJECT GpuIncoreWorkload.cu)

target_include_directories(gpu-incore-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-incore-cuda-obj)
target_link_libraries(gpu-incore-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-incore-cuda $<TARGET_OBJECTS:gpu-incore-cuda-obj>)
    target_link_libraries(gpu-incore-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-incore-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

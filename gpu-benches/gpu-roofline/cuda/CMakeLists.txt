cmake_minimum_required(VERSION 3.15)

add_library(gpu-roofline-cuda-obj OBJECT GpuRooflineWorkload.cu)

target_include_directories(gpu-roofline-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-roofline-cuda-obj)
target_link_libraries(gpu-roofline-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-roofline-cuda $<TARGET_OBJECTS:gpu-roofline-cuda-obj>)
    target_link_libraries(gpu-roofline-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-roofline-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

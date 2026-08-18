cmake_minimum_required(VERSION 3.15)

add_library(gpu-l2-stream-hipifiable-obj OBJECT GpuL2StreamWorkload.hip)

target_include_directories(gpu-l2-stream-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-l2-stream-hipifiable-obj)
target_link_libraries(gpu-l2-stream-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-l2-stream-hipifiable $<TARGET_OBJECTS:gpu-l2-stream-hipifiable-obj>)
    target_link_libraries(gpu-l2-stream-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-l2-stream-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

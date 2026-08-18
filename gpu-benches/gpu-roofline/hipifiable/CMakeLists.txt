cmake_minimum_required(VERSION 3.15)

add_library(gpu-roofline-hipifiable-obj OBJECT GpuRooflineWorkload.hip)

target_include_directories(gpu-roofline-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-roofline-hipifiable-obj)
target_link_libraries(gpu-roofline-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-roofline-hipifiable $<TARGET_OBJECTS:gpu-roofline-hipifiable-obj>)
    target_link_libraries(gpu-roofline-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-roofline-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

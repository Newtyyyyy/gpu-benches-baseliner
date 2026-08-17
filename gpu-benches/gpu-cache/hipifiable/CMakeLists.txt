cmake_minimum_required(VERSION 3.15)

add_library(gpu-cache-hipifiable-obj OBJECT GpuCacheWorkload.hip)

target_include_directories(gpu-cache-hipifiable-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-cache-hipifiable-obj)
target_link_libraries(gpu-cache-hipifiable-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-cache-hipifiable $<TARGET_OBJECTS:gpu-cache-hipifiable-obj>)
    target_link_libraries(gpu-cache-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-cache-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

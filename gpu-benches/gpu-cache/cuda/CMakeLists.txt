cmake_minimum_required(VERSION 3.15)

add_library(gpu-cache-cuda-obj OBJECT gpu-cache.cu)

target_include_directories(gpu-cache-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-cache-cuda-obj)
target_link_libraries(gpu-cache-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(gpu-cache-cuda $<TARGET_OBJECTS:gpu-cache-cuda-obj>)
    target_link_libraries(gpu-cache-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-cache-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
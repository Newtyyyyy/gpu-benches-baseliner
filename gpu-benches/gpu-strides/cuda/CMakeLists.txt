cmake_minimum_required(VERSION 3.15)

add_library(gpu-strides-cuda-obj OBJECT GpuStridesWorkload.cu)

target_include_directories(gpu-strides-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-strides-cuda-obj)
target_link_libraries(gpu-strides-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

# CUPTI/Perfworks, for the L1/L2 hardware-counter metrics
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu-strides-cuda-obj PUBLIC
    CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target CUDA::cuda_driver)

# CUPTI engine compiled into the native CUDA build only
target_compile_definitions(gpu-strides-cuda-obj PRIVATE STRIDES_ENABLE_CUPTI)

# cupti/ SCOPE_EXIT macros trip -Wunused-value, which -Werror would reject
target_compile_options(gpu-strides-cuda-obj PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=unused-value>")

if(NOT COMBINED_BUILD)
    add_executable(gpu-strides-cuda $<TARGET_OBJECTS:gpu-strides-cuda-obj>)
    target_link_libraries(gpu-strides-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-strides-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

cmake_minimum_required(VERSION 3.15)

add_library(gpu-strides-cuda-obj OBJECT GpuStridesWorkload.cu)

target_include_directories(gpu-strides-cuda-obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS gpu-strides-cuda-obj)
target_link_libraries(gpu-strides-cuda-obj PUBLIC baseliner::baseliner gpu-benches-baseliner)

# CUPTI / Perfworks for the L1/L2 hardware-counter metrics (cupti/ engine,
# ported from RRZE gpu-metrics). Imported targets carry include + lib paths.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu-strides-cuda-obj PUBLIC
    CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target CUDA::cuda_driver)

# CUPTI L1/L2 counters are compiled in only for the native CUDA build. The
# mechanically-hipified variant (hipifiable/) is built without this define,
# so it excludes the CUPTI engine and reports 0 for those counters.
target_compile_definitions(gpu-strides-cuda-obj PRIVATE STRIDES_ENABLE_CUPTI)

# The ported CUPTI engine (cupti/) uses SCOPE_EXIT RAII macros that trip
# -Wunused-value; the baseliner builds with -Werror. Relax just that warning
# for this object (our own code doesn't rely on it).
target_compile_options(gpu-strides-cuda-obj PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=unused-value>")

if(NOT COMBINED_BUILD)
    add_executable(gpu-strides-cuda $<TARGET_OBJECTS:gpu-strides-cuda-obj>)
    target_link_libraries(gpu-strides-cuda PRIVATE baseliner::baseliner gpu-benches-baseliner)
    set_target_properties(gpu-strides-cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

endif()

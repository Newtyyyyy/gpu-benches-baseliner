cmake_minimum_required(VERSION 3.15)
project(gpu-benches-baseliner VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FOUND_DEVICE_COMPILER OFF)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    set(HAS_CUDA ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(HAS_HIP ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

if(NOT FOUND_DEVICE_COMPILER)
    message(FATAL_ERROR "Requires at least one device device compiler")
endif()

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v1.0
)

FetchContent_MakeAvailable(baseliner)
include(${baseliner_SOURCE_DIR}/cmake/utils.cmake)


#TEMPORARY FIX AS THROUGH PRESETS IT DOES NOT WORK
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Create a "version" interface
add_library(gpu-benches-baseliner INTERFACE)

# Run your function on this interface
baseliner_enable_git_version(gpu-benches-baseliner)
option(COMBINED_BUILD "Build everything into one executable" ON)
option(BASELINER_BUILD_HIPIFIABLE "Build the mechanically-hipified (hipify-perl) variant instead of the hand-tuned hip/ one" OFF)
option(BASELINER_FAST_MATH "Enable fast-math for the CUDA and hipifiable benchmark kernels (never the hand-tuned hip/ reference)" ON)

set_property(GLOBAL PROPERTY APP_OBJECT_TARGETS "")

add_subdirectory(gpu-benches)

if(BASELINER_FAST_MATH)
    # Apply fast-math only to the cuda/ (.cu) and hipifiable/ (.hip) benchmark object
    # targets, matched by their naming convention. The hand-tuned hip/ reference targets
    # (-hip-obj) are deliberately left untouched so they stay bit-for-bit as authored.
    #
    # The right flag depends on what actually compiles the HIP sources: nvcc under HIP's
    # NVIDIA platform wants --use_fast_math, an amdclang++/ROCm compiler wants -ffast-math.
    if(CMAKE_HIP_COMPILER MATCHES "nvcc")
        set(_hipifiable_fast_math --use_fast_math)
    else()
        set(_hipifiable_fast_math -ffast-math)
    endif()

    get_property(_fm_obj_targets GLOBAL PROPERTY APP_OBJECT_TARGETS)
    foreach(_fm_target IN LISTS _fm_obj_targets)
        if(_fm_target MATCHES "-cuda-obj$")
            target_compile_options(${_fm_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        elseif(_fm_target MATCHES "-hipifiable-obj$")
            target_compile_options(${_fm_target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${_hipifiable_fast_math}>)
        endif()
    endforeach()
endif()

if(COMBINED_BUILD)
    get_property(ALL_OBJ_TARGETS GLOBAL PROPERTY APP_OBJECT_TARGETS)
    
    add_executable(gpu-benches_exec)
    
    foreach(obj_target IN LISTS ALL_OBJ_TARGETS)
        target_sources(gpu-benches_exec PRIVATE $<TARGET_OBJECTS:${obj_target}>)
    endforeach()

    target_link_libraries(gpu-benches_exec PRIVATE baseliner::baseliner gpu-benches-baseliner)

    # gpu-strides measures L1/L2 hardware counters via CUPTI/Perfworks. The
    # combined exec pulls in object files but not their usage requirements, so
    # link the profiler libs here where the executable is created.
    if(HAS_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(gpu-benches_exec PRIVATE
            CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target CUDA::cuda_driver)
    endif()

    # HIP-on-NVIDIA ONLY (not a real AMD build). Both cases are "HAS_HIP AND NOT
    # HAS_CUDA", so gate on the HIP platform: nvidia => HipBackend aliases
    # CudaBackend and hardware stats come from NVML via this shim; amd => the
    # real AMD-SMI path in baseliner-src is used and this shim must NOT compile
    # (it pulls CUDA/NVML headers that don't exist on an AMD node and forces the
    # NVIDIA HIP platform, breaking hip_runtime.h).
    if(HAS_HIP AND NOT HAS_CUDA AND CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        # HipBackend == CudaBackend under HIP's NVIDIA platform (hipStream_t is a cudaStream_t
        # typedef), so baseliner-src never defines ClockFrequency/DeviceTemperature/
        # DevicePowerUtilization<HipBackend> here (that only happens via AMD-SMI, which this
        # node doesn't have). Provide the same specializations backed by NVML instead.
        find_package(CUDAToolkit REQUIRED)
        target_sources(gpu-benches_exec PRIVATE gpu-benches/build_hipcuda/hip_nvidia_stats_shim.cpp)
        target_link_libraries(gpu-benches_exec PRIVATE CUDA::cudart CUDA::nvml)
        if(DEFINED ENV{HIP_PATH})
            set_source_files_properties(gpu-benches/build_hipcuda/hip_nvidia_stats_shim.cpp PROPERTIES
                COMPILE_OPTIONS "-isystem;$ENV{HIP_PATH}/include;-D__HIP_PLATFORM_NVIDIA__;-D__HIP_PLATFORM_NVCC__")
        endif()
    endif()
endif()
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
    # Fast-math for the cuda/ and hipifiable/ objects only, never the hand-tuned hip/ ones
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

    # CUPTI libs for gpu-strides: object usage requirements don't reach the combined exec
    if(HAS_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(gpu-benches_exec PRIVATE
            CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target CUDA::cuda_driver)
    endif()

    # HIP-on-NVIDIA only: NVML-backed stats shim, since AMD-SMI isn't available there
    if(HAS_HIP AND NOT HAS_CUDA AND CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        find_package(CUDAToolkit REQUIRED)
        target_sources(gpu-benches_exec PRIVATE gpu-benches/build_hipcuda/hip_nvidia_stats_shim.cpp)
        target_link_libraries(gpu-benches_exec PRIVATE CUDA::cudart CUDA::nvml)
        if(DEFINED ENV{HIP_PATH})
            set_source_files_properties(gpu-benches/build_hipcuda/hip_nvidia_stats_shim.cpp PROPERTIES
                COMPILE_OPTIONS "-isystem;$ENV{HIP_PATH}/include;-D__HIP_PLATFORM_NVIDIA__;-D__HIP_PLATFORM_NVCC__")
        endif()
    endif()
endif()
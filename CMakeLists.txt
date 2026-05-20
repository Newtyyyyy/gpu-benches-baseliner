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

set_property(GLOBAL PROPERTY APP_OBJECT_TARGETS "")

add_subdirectory(gpu-benches)

if(COMBINED_BUILD)
    get_property(ALL_OBJ_TARGETS GLOBAL PROPERTY APP_OBJECT_TARGETS)
    
    add_executable(gpu-benches_exec)
    
    foreach(obj_target IN LISTS ALL_OBJ_TARGETS)
        target_sources(gpu-benches_exec PRIVATE $<TARGET_OBJECTS:${obj_target}>)
    endforeach()

    target_link_libraries(gpu-benches_exec PRIVATE baseliner::baseliner gpu-benches-baseliner)
endif()
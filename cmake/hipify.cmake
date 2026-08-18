# add_hipified_bench(<bench-name> <cuda-source>)
#
# Generates the mechanically-hipified variant of a benchmark at build time and
# declares its object target, replacing the former committed hipifiable/ trees.
# <cuda-source> is relative to the calling CMakeLists (e.g. cuda/GpuXWorkload.cu).

set(HIPIFY_ONE_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/../gpu-benches/build_hipcuda/hipify_one.sh)

function(add_hipified_bench BENCH CUDA_SOURCE)
    find_program(HIPIFY_PERL hipify-perl)
    if(NOT HIPIFY_PERL)
        message(FATAL_ERROR "BASELINER_BUILD_HIPIFIABLE=ON needs hipify-perl in PATH (ships with ROCm)")
    endif()

    set(source ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE})
    get_filename_component(stem ${CUDA_SOURCE} NAME_WE)
    set(generated ${CMAKE_CURRENT_BINARY_DIR}/hipifiable/${stem}.hip)

    add_custom_command(
        OUTPUT ${generated}
        COMMAND bash ${HIPIFY_ONE_SCRIPT} ${source} ${generated}
        DEPENDS ${source} ${HIPIFY_ONE_SCRIPT}
        COMMENT "hipify ${BENCH}"
        VERBATIM)

    set(target ${BENCH}-hipifiable-obj)
    add_library(${target} OBJECT ${generated})
    set_source_files_properties(${generated} PROPERTIES LANGUAGE HIP)

    # The generated file lives in the build tree, so the include path has to point
    # back at the source tree for its "../<Bench>Workload.hpp" include to resolve.
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PUBLIC baseliner::baseliner gpu-benches-baseliner)
    set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS ${target})

    if(NOT COMBINED_BUILD)
        add_executable(${BENCH}-hipifiable $<TARGET_OBJECTS:${target}>)
        target_link_libraries(${BENCH}-hipifiable PRIVATE baseliner::baseliner gpu-benches-baseliner)
        set_target_properties(${BENCH}-hipifiable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()
endfunction()

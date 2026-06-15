if(HAS_CUDA)
add_subdirectory(cuda)
endif()

if(HAS_HIP)
add_subdirectory(hip)
endif()

if(HAS_NVBENCH_ADAPTER)
add_subdirectory(workload)
endif()
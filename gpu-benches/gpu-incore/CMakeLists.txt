if(HAS_CUDA)
add_subdirectory(cuda)
endif()

if(HAS_HIP)
add_subdirectory(hip)
endif()
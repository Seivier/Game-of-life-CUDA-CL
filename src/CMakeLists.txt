add_subdirectory(cpu)
add_subdirectory(cl)
# if (CMAKE_CUDA_COMPILER)
#    message(STATUS "CUDA compiler found, enabling CUDA support")
#    enable_language(CUDA)
add_subdirectory(cuda)
# endif ()
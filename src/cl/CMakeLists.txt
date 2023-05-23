find_package(OpenCL REQUIRED)
add_executable(${PROJECT_NAME}-CL "main.cpp")
target_link_libraries(${PROJECT_NAME}-CL PRIVATE OpenCL::OpenCL)
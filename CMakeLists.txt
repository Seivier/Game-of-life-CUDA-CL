cmake_minimum_required(VERSION 3.0)
project(Game-of-life LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 61)
add_subdirectory(src)

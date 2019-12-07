cmake_minimum_required(VERSION 3.10)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 14)

add_executable(ParallelProgramming main.cpp)
SET(CMAKE_CXX_FLAGS "-Wall -fno-rtti -O2")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelProgramming PUBLIC OpenMP::OpenMP_CXX)
endif()
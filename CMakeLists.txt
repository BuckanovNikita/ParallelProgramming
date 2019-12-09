cmake_minimum_required(VERSION 3.10)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 14)

add_executable(ParallelProgramming main.cpp)
add_executable(ParallelProgramming2 integrate.cpp integrate.cpp)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O0 ")
SET(CMAKE_CXX_FLAGS "-Ofast")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelProgramming PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(ParallelProgramming2 PUBLIC OpenMP::OpenMP_CXX)
endif()
project(openpifpaf_postprocessing)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-O3 -std=c++17")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SOURCE_MAIN "src/*.cpp" "test/main.cpp")

add_executable(openpifpaf_postprocessor_main ${SOURCE_MAIN})
target_include_directories(openpifpaf_postprocessor_main INTERFACE ./include)

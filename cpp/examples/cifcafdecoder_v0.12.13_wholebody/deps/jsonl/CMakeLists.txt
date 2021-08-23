cmake_minimum_required(VERSION 3.10)

project(jsonl)

add_library(jsonl INTERFACE)
add_library(jsonl::library ALIAS jsonl)
target_include_directories(jsonl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(decoder-example)

# set(CMAKE_CXX_STANDARD 11)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SRC *.cpp)
add_executable(decoder-example ${SRC})
target_link_libraries(decoder-example "${TORCH_LIBRARIES}")
set_property(TARGET decoder-example PROPERTY CXX_STANDARD 17)

# set(BIN "decoder-example")

# add_executable(${BIN} ${SRC})


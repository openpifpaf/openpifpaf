set(CMAKE_CXX_STANDARD 11)

set(BIN "decoder-example")
file(GLOB SRC *.cpp)

add_executable(${BIN} ${SRC})


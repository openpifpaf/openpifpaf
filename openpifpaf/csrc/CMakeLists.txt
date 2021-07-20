cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(openpifpaf-example)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB SRC src/*.cpp)
include_directories(include/)
add_executable(openpifpaf-example main.cpp ${SRC})
target_link_libraries(openpifpaf-example "${TORCH_LIBRARIES}")
target_link_libraries(openpifpaf-example "${OpenCV_LIBS}")

# for additional debug info:
target_compile_definitions(openpifpaf-example PRIVATE DEBUG=1)

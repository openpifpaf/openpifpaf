cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(openpifpaf-examples)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
message(STATUS "torch cxx flags: ${TORCH_CXX_FLAGS}")
message(STATUS "torch libraries: ${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV COMPONENTS core imgproc highgui imgcodecs videoio REQUIRED)
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libraries: ${OpenCV_LIBS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB SRC ${CMAKE_SOURCE_DIR}/../openpifpaf/csrc/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/../openpifpaf/csrc/include/)


### for static library
add_library(openpifpafstatic STATIC ${SRC})
target_link_libraries(openpifpafstatic "${TORCH_LIBRARIES}")


### for shared library
add_library(openpifpaf SHARED ${SRC})
target_link_libraries(openpifpaf "${TORCH_LIBRARIES}")


### for openpifpaf-image
add_executable(openpifpaf-image cli_image.cpp ${SRC})
target_link_libraries(openpifpaf-image "${TORCH_LIBRARIES}")
target_link_libraries(openpifpaf-image "${OpenCV_LIBS}")
# for additional debug info:
target_compile_definitions(openpifpaf-image PRIVATE DEBUG=1)


### for openpifpaf-video
add_executable(openpifpaf-video cli_video.cpp ${SRC})
target_link_libraries(openpifpaf-video "${TORCH_LIBRARIES}")
target_link_libraries(openpifpaf-video "${OpenCV_LIBS}")
# for additional debug info:
target_compile_definitions(openpifpaf-video PRIVATE DEBUG=1)

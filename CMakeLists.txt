cmake_minimum_required(VERSION 3.10)
project(svm)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
add_executable(svm main.cpp)

target_link_libraries(svm ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.16)
project(RM_svm)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
add_executable(RM_svm main.cpp)

target_link_libraries(RM_svm ${OpenCV_LIBS})
cmake_minimum_required(VERSION 2.8)
project(semantic_nav)
add_definitions("-std=gnu++11")
#include_directories(include)
find_package(OpenCV REQUIRED)

add_executable(semantic_main src/semantic_main.cpp)
add_executable(watershed_test src/watershed_test.cpp)
add_executable(histogram_test src/histogram_test.cpp)

target_link_libraries(semantic_main ${OpenCV_LIBS})
target_link_libraries(watershed_test ${OpenCV_LIBS})
target_link_libraries(histogram_test ${OpenCV_LIBS})

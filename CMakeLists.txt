cmake_minimum_required(VERSION 3.14)
project(pcl_test)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL 1.2 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})

add_executable(pcl_test main.cpp)

target_link_libraries(pcl_test ${PCL_LIBRARIES})

cmake_minimum_required(VERSION 3.16)
project(test_memcpy)
set(CMAKE_CXX_STANDARD 14)

add_executable(test_memcpy main.cpp)
target_link_libraries(test_memcpy pthread)

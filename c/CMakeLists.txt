cmake_minimum_required(VERSION 3.8)
project(c)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES c_api.cc c_api.h c_api_internal.h)
add_library(c ${SOURCE_FILES})
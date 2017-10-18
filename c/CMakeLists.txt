cmake_minimum_required(VERSION 2.8)
project(dynetc)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror")

find_package(Eigen3 REQUIRED)
find_package(DyNet REQUIRED)

# set(DYNET_LIBRARIES "dynet")

include_directories(
    ${PROJECT_SOURCE_DIR}
    # ${PROJECT_BINARY_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${DYNET_INCLUDE_DIR}
)

link_directories(
    ${DYNET_LIBRARY_DIR}
)

add_subdirectory(dynetc)

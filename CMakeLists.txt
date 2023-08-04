cmake_minimum_required(VERSION 3.16)
project(flow_lines)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

igl_include(embree)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::embree)

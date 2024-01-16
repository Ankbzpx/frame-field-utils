cmake_minimum_required(VERSION 3.16)
project(frame_field_utils_bind)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)
igl_include(embree)
igl_include(copyleft comiso)

include(scs)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC igl::embree igl_copyleft::comiso scsdir)

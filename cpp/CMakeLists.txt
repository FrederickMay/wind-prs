cmake_minimum_required(VERSION 3.20)
project(prs_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

pybind11_add_module(_engine src/engine.cpp src/bindings.cpp)
target_link_libraries(_engine PRIVATE Eigen3::Eigen)
if(MSVC)
  target_compile_options(_engine PRIVATE /O2 /W4 /permissive-)
else()
  target_compile_options(_engine PRIVATE -O3 -Wall -Wextra)
endif()

# put the built module next to the Python package so `from prs import _engine` works
set_target_properties(_engine PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/prs
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../src/prs)

cmake_minimum_required(VERSION 3.15)
project(evolib)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

file(GLOB CU_SOURCES src/util/*.cu)
file(GLOB CPP_SOURCES src/util/*.cpp)

add_library(evallib STATIC ${CPP_SOURCES})
target_include_directories(evallib PUBLIC include/util)

nanobind_add_module(evoapp src/python/bindings.cpp)

target_link_libraries(evoapp PUBLIC evallib)

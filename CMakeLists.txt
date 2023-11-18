cmake_minimum_required(VERSION 3.24)
project(evolib)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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

# Optional CUDA linkage

include(CheckLanguage)
check_language(CUDA)

if (DEFINED CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    add_library(evallibcuda STATIC ${CU_SOURCES})
    target_include_directories(evallibcuda PUBLIC include/util)
    set_target_properties(evallibcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(evallibcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(evallibcuda PROPERTIES CUDA_ARCHITECTURES native)
    set_target_properties(evoapp PROPERTIES CUDA_ARCHITECTURES native)

    target_link_libraries(evoapp PUBLIC evallibcuda)
endif()

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)

project(TrianglesProject LANGUAGES CXX)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(APPLE)
  set(USE_CUDA OFF)
endif()

if(USE_CUDA)
  message("Enabling CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "89")
  find_package(CUDAToolkit REQUIRED)
  find_package(Thrust REQUIRED)
  thrust_create_target(Thrust)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# enable AddressSanitizer if in debug mode
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   message("Enabling AddressSanitizer")
#   add_compile_options(-fsanitize=address)
#   add_link_options(-fsanitize=address)
# endif()

# if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
#   message("Enabling no-omit-frame-pointer")
#   add_compile_options(-fno-omit-frame-pointer)
#   add_link_options(-fno-omit-frame-pointer)
# endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Enabling Release optimizations")
  add_compile_options(-O3 -march=native -mtune=native -fassociative-math)
  add_link_options(-O3 -march=native -mtune=native -fassociative-math)
endif()

add_compile_options(-mavx512f -mavx512vl -mavx512dq -mavx512bw -mavx512cd)

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect headers for the core library
file(GLOB_RECURSE CORE_HEADER_FILES "src/*.h")
file(GLOB_RECURSE CORE_IMPLEMENTATION_FILES "src/*.cpp")

add_library(TrianglesProjectCore ${CORE_HEADER_FILES} ${CORE_IMPLEMENTATION_FILES})
target_include_directories(TrianglesProjectCore PUBLIC "src/")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# include(CheckIPOSupported)
# check_ipo_supported(RESULT supported OUTPUT error)

# if( supported )
#   message(STATUS "IPO / LTO is supported on this platform.")
#   add_compile_options(-flto=auto) # -fno-fat-lto-objects
#   add_link_options(-flto=auto)
# else()
#   message(STATUS "IPO / LTO not supported: <${error}>")
# endif()

if(USE_CUDA)
  target_link_libraries(TrianglesProjectCore INTERFACE CUDA::cublas CUDA::cudart)
  target_link_libraries(TrianglesProjectCore INTERFACE Thrust)
endif()

# LodePNG dependency
add_library(LodePNG STATIC "ext/lodepng/lodepng.cpp" "ext/lodepng/lodepng.h")
target_include_directories(LodePNG PUBLIC "ext/lodepng/")
target_link_libraries(TrianglesProjectCore INTERFACE LodePNG)

# JSON dependency
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(TrianglesProjectCore INTERFACE nlohmann_json::nlohmann_json)

# OpenGL dependency
find_package(OpenGL REQUIRED)
target_link_libraries(TrianglesProjectCore INTERFACE OpenGL::GL OpenGL::GLU)
# GLM dependency
find_package(glm REQUIRED)
target_link_libraries(TrianglesProjectCore INTERFACE glm::glm)
# GLFW dependency
find_package(glfw3 REQUIRED)
target_link_libraries(TrianglesProjectCore INTERFACE glfw)
# GLEW dependency
find_package(GLEW REQUIRED)
target_link_libraries(TrianglesProjectCore INTERFACE GLEW::GLEW)
# Cairo dependency
find_package(Cairo REQUIRED)
target_link_libraries(TrianglesProjectCore INTERFACE cairo)
target_link_libraries(TrianglesProjectCore INTERFACE ${Cairo_LIBRARIES})
target_include_directories(TrianglesProjectCore INTERFACE ${Cairo_INCLUDE_DIRS})
# GoogleTest dependency
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
  )
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
# Initialize GoogleTest
enable_testing()
# GoogleBenchmark dependency
add_subdirectory(ext/benchmark)

# For each subfolder in the "experiments/" folder, create a separate executable
file(GLOB EXPERIMENTS_DIRS "experiments/*")
foreach(EXPERIMENT_DIR ${EXPERIMENTS_DIRS})
  if(IS_DIRECTORY ${EXPERIMENT_DIR})
    get_filename_component(EXPERIMENT_NAME ${EXPERIMENT_DIR} NAME)
    file(GLOB_RECURSE EXPERIMENT_IMPL_HEADERS "${EXPERIMENT_DIR}/*.h")
    file(GLOB_RECURSE EXPERIMENT_TEST_FILES "${EXPERIMENT_DIR}/test*.cpp")
    file(GLOB_RECURSE EXPERIMENT_BENCH_FILES "${EXPERIMENT_DIR}/bench*.cpp")

    # create test executable and benchmark executable
    add_executable(${EXPERIMENT_NAME}_test ${EXPERIMENT_TEST_FILES} ${EXPERIMENT_IMPL_HEADERS})
    target_link_libraries(${EXPERIMENT_NAME}_test GTest::gtest_main TrianglesProjectCore)
    target_include_directories(${EXPERIMENT_NAME}_test PUBLIC ${EXPERIMENT_DIR})
    gtest_discover_tests(${EXPERIMENT_NAME}_test)

    add_executable(${EXPERIMENT_NAME}_bench ${EXPERIMENT_BENCH_FILES} ${EXPERIMENT_IMPL_HEADERS})
    target_link_libraries(${EXPERIMENT_NAME}_bench TrianglesProjectCore)
    target_link_libraries(${EXPERIMENT_NAME}_bench benchmark::benchmark)
    target_link_libraries(${EXPERIMENT_NAME}_bench benchmark::benchmark_main)
    target_include_directories(${EXPERIMENT_NAME}_bench PUBLIC ${EXPERIMENT_DIR})
  endif()
endforeach()

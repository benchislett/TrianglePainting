cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 23)

project(PolyPaintLib LANGUAGES CXX)

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
  # find_package(Thrust REQUIRED)
  # thrust_create_target(Thrust)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# enable AddressSanitizer if in debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Enabling AddressSanitizer")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Enabling no-omit-frame-pointer")
  add_compile_options(-fno-omit-frame-pointer)
  add_link_options(-fno-omit-frame-pointer)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Enabling Release optimizations")
  add_compile_options(-O3 -march=native -mtune=native -ffast-math)
  add_link_options(-O3 -march=native -mtune=native -ffast-math)
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
file(GLOB_RECURSE HEADERS "include/*.h")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_HEADERS "include/*.cuh")
  list(APPEND HEADERS ${CUDA_HEADERS})
endif()

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
  message(STATUS "IPO / LTO is supported on this platform.")
  add_compile_options(-flto=auto -fno-fat-lto-objects )
  add_link_options(-flto=auto -fno-fat-lto-objects)
else()
  message(STATUS "IPO / LTO not supported: <${error}>")
endif()

# Define PolyPaintLib as a header-only library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)

if(USE_CUDA)
  target_link_libraries(${PROJECT_NAME} INTERFACE CUDA::cublas CUDA::cudart)
  # target_link_libraries(${PROJECT_NAME} INTERFACE Thrust)
endif()

# LodePNG dependency
add_library(LodePNG STATIC "ext/lodepng/lodepng.cpp" "ext/lodepng/lodepng.h")
target_include_directories(LodePNG PUBLIC "ext/lodepng/")
target_link_libraries(${PROJECT_NAME} INTERFACE LodePNG)

# JSON dependency
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(${PROJECT_NAME} INTERFACE nlohmann_json::nlohmann_json)

# OpenGL dependency
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenGL::GL OpenGL::GLU)
# GLM dependency
find_package(glm REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE glm::glm)
# GLFW dependency
find_package(glfw3 REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE glfw)
# GLEW dependency
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE GLEW::GLEW)

# set up GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# load googletest tests
enable_testing()

# Collect test sources
file(GLOB_RECURSE TESTS "tests_cpp/*.cpp")
add_executable(Test ${TESTS})
target_link_libraries(Test GTest::gtest_main ${PROJECT_NAME})

include(GoogleTest)
gtest_discover_tests(Test)

find_package(Cairo REQUIRED)

# set up experiments
# create an executable for each .cpp source file in folder "experiments"
file(GLOB_RECURSE EXPERIMENTS "experiments/*.cpp")
file(GLOB_RECURSE EXPERIMENT_HEADERS "experiments/*.h")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_EXPERIMENTS "experiments/*.cu")
  list(APPEND EXPERIMENTS ${CUDA_EXPERIMENTS})
endif()

foreach(EXPERIMENT ${EXPERIMENTS})
  get_filename_component(EXPERIMENT_NAME ${EXPERIMENT} NAME_WE)
  add_executable(${EXPERIMENT_NAME} ${EXPERIMENT} ${EXPERIMENT_HEADERS})
  target_link_libraries(${EXPERIMENT_NAME} ${PROJECT_NAME})
  target_include_directories(${EXPERIMENT_NAME} PUBLIC experiments)
  target_link_libraries(${EXPERIMENT_NAME} ${Cairo_LIBRARIES})
  target_include_directories(${EXPERIMENT_NAME} PUBLIC ${Cairo_INCLUDE_DIRS})
  target_link_libraries(${EXPERIMENT_NAME} cairo)
endforeach()

add_subdirectory(ext/benchmark)
file(GLOB_RECURSE BENCHMARKS "benchmarks/*.cpp")
file(GLOB_RECURSE BENCHMARK_HEADERS "benchmarks/*.h")
add_executable(Benchmark ${BENCHMARKS} ${BENCHMARK_HEADERS})
target_link_libraries(Benchmark benchmark::benchmark ${PROJECT_NAME})
target_link_libraries(Benchmark benchmark::benchmark_main ${PROJECT_NAME})
target_include_directories(Benchmark PUBLIC benchmarks)

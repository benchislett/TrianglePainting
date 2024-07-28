cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

project(Science LANGUAGES CXX)

option(USE_CUDA "Use CUDA" OFF)

if(USE_CUDA)
  message("Enabling CUDA")
  enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "89")
  find_package(Thrust REQUIRED CONFIG)
  thrust_create_target(Thrust)
  find_package(CUDAToolkit REQUIRED)
endif()

# enable AddressSanitizer if in debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Enabling AddressSanitizer")
  add_compile_options(-fsanitize=address)
  add_compile_options(-fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
  add_link_options(-fno-omit-frame-pointer)
endif()

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")
file(GLOB_RECURSE TESTS "test/*.cpp")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
  file(GLOB_RECURSE CUDA_HEADERS "include/*.cuh")
  list(APPEND SOURCES ${CUDA_SOURCES})
  list(APPEND HEADERS ${CUDA_HEADERS})
endif()

add_library(Science STATIC ${SOURCES} ${HEADERS})

if(USE_CUDA)
  target_link_libraries(Science PUBLIC CUDA::cublas)
  target_link_libraries(Science PUBLIC Thrust)
endif()

target_include_directories(Science PUBLIC include)

# LodePNG dependency
add_library(LodePNG STATIC "ext/lodepng/lodepng.cpp" "ext/lodepng/lodepng.h")
target_include_directories(LodePNG PUBLIC "ext/lodepng/")
target_link_libraries(Science PUBLIC LodePNG)

# JSON dependency
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(Science PUBLIC nlohmann_json::nlohmann_json)

# set up GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
  FIND_PACKAGE_ARGS NAMES GTest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# load googletest tests
enable_testing()

add_executable(Test ${TESTS})
target_link_libraries(Test GTest::gtest_main Science)

include(GoogleTest)
gtest_discover_tests(Test)

# set up experiments
# create an executable for each .cpp source file in folder "experiments"
file(GLOB_RECURSE EXPERIMENTS "experiments/*.cpp")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_EXPERIMENTS "experiments/*.cu")
  list(APPEND EXPERIMENTS ${CUDA_EXPERIMENTS})
endif()

foreach(EXPERIMENT ${EXPERIMENTS})
  get_filename_component(EXPERIMENT_NAME ${EXPERIMENT} NAME_WE)
  add_executable(${EXPERIMENT_NAME} ${EXPERIMENT})
  target_link_libraries(${EXPERIMENT_NAME} Science)
endforeach()
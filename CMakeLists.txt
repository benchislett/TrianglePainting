cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

project(Science)

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

add_library(Science STATIC ${SOURCES} ${HEADERS})
target_include_directories(Science PUBLIC include)

# LodePNG dependency
add_library(LodePNG STATIC "ext/lodepng/lodepng.cpp" "ext/lodepng/lodepng.h")
target_include_directories(LodePNG PUBLIC "ext/lodepng/")
target_link_libraries(Science PUBLIC LodePNG)

# set up CUDA as an optional dependency
option(USE_CUDA "Use CUDA" OFF)
if(USE_CUDA)
  enable_language(CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
  add_library(Science_CUDA STATIC ${CUDA_SOURCES})
  target_include_directories(Science_CUDA PUBLIC include)
  target_link_libraries(Science_CUDA PUBLIC Science)
endif()

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
foreach(EXPERIMENT ${EXPERIMENTS})
  get_filename_component(EXPERIMENT_NAME ${EXPERIMENT} NAME_WE)
  add_executable(${EXPERIMENT_NAME} ${EXPERIMENT})
  target_link_libraries(${EXPERIMENT_NAME} Science)
endforeach()
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20)

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

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")
file(GLOB_RECURSE TESTS "tests_cpp/*.cpp")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
  file(GLOB_RECURSE CUDA_HEADERS "include/*.cuh")
  list(APPEND SOURCES ${CUDA_SOURCES})
  list(APPEND HEADERS ${CUDA_HEADERS})
endif()

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

add_compile_options(-mavx2 -msse4.2)
if( supported )
  message(STATUS "IPO / LTO is supported on this platform.")
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  add_compile_options(-O3 -flto=auto -fno-fat-lto-objects -march=native -mtune=native)
  add_link_options(-O3 -flto=auto -fno-fat-lto-objects -march=native -mtune=native)
else()
  message(STATUS "IPO / LTO not supported: <${error}>")
endif()

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

if(USE_CUDA)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
  # target_link_libraries(${PROJECT_NAME} PUBLIC Thrust)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC include)

# LodePNG dependency
add_library(LodePNG STATIC "ext/lodepng/lodepng.cpp" "ext/lodepng/lodepng.h")
target_include_directories(LodePNG PUBLIC "ext/lodepng/")
target_link_libraries(${PROJECT_NAME} PUBLIC LodePNG)

# JSON dependency
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

# OpenGL dependency
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GLU)
# GLM dependency
find_package(glm REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC glm::glm)
# GLFW dependency
find_package(glfw3 REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
# GLEW dependency
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC GLEW::GLEW)

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

# set up python bindings
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Detect the installed nanobind package and import it into CMake
message( "${Python_EXECUTABLE}" )
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# add the Python bindings
file(GLOB_RECURSE PYBINDINGS "bindings/*.cpp")
file(GLOB_RECURSE PYHEADERS "bindings/*.h")
nanobind_add_module(polypaint STABLE_ABI NB_STATIC ${PYBINDINGS} ${PYHEADERS})
target_link_libraries(polypaint PRIVATE ${PROJECT_NAME})
target_include_directories(polypaint PRIVATE bindings)
install(TARGETS polypaint LIBRARY DESTINATION polypaint)

add_subdirectory(ext/benchmark)
file(GLOB_RECURSE BENCHMARKS "benchmarks/*.cpp")
file(GLOB_RECURSE BENCHMARK_HEADERS "benchmarks/*.h")
add_executable(Benchmark ${BENCHMARKS} ${BENCHMARK_HEADERS})
target_link_libraries(Benchmark benchmark::benchmark ${PROJECT_NAME})
target_link_libraries(Benchmark benchmark::benchmark_main ${PROJECT_NAME})
target_include_directories(Benchmark PUBLIC benchmarks)

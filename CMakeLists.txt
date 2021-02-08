cmake_minimum_required(VERSION 3.14...3.18)

project(
  ekf
  VERSION 0.2
  DESCRIPTION "A very simple yet powerful ekf"
  LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   # need this for concepts
   set(CMAKE_CXX_STANDARD 20)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   # Let's ensure -std=c++xx instead of -std=g++xx
   set(CMAKE_CXX_EXTENSIONS OFF)
   set_property(GLOBAL PROPERTY USE_FOLDERS ON)

   include(CTest)

endif()

find_package(Eigen3 REQUIRED)
find_package(autodiff CONFIG REQUIRED) 

add_library(ekf INTERFACE)
target_sources(ekf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/ekf.h)
target_include_directories(ekf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/) 
target_link_libraries(ekf INTERFACE Eigen3::Eigen autodiff::autodiff)

# build examples
option(EKF_BUILD_EXAMPLE "build examples" OFF)
if(${EKF_BUILD_EXAMPLE} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(example/)
endif()

# add testing
option(EKF_BUILD_TEST "build tests" OFF)
if(${EKF_BUILD_TEST} OR (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    enable_testing()
    add_subdirectory(test/)
endif()

# TODO : install stuff
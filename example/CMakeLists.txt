# fetch external matplotlib-cpp module for visualization needed in the examples
include(FetchContent)
FetchContent_Declare(
  matplotlib-cpp
  GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/matplotlib-cpp
)
FetchContent_MakeAvailable(matplotlib-cpp)

# needed as well for visualization
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

add_definitions(-DEXAMPLES_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")

add_subdirectory(lidar_sensor)
add_subdirectory(radar_sensor)
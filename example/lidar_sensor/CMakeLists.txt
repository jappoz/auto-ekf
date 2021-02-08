
add_executable(lidar_example main.cc)

target_include_directories(lidar_example PUBLIC ${matplotlib-cpp_SOURCE_DIR})
target_link_libraries(lidar_example  PUBLIC
  Python3::Python 
  Python3::NumPy
  ekf)
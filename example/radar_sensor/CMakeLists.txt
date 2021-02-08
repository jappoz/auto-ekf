
add_executable(radar_example main.cc)

target_include_directories(radar_example PUBLIC ${matplotlib-cpp_SOURCE_DIR})
target_link_libraries(radar_example  PUBLIC
  Python3::Python 
  Python3::NumPy
  ekf)
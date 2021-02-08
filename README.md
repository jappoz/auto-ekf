# auto-ekf

auto-ekf is a super simple header-only Extended Kalman Filter class making use of automatic differentiation for system linearization [(from this awesome repo)](https://github.com/autodiff/autodiff) and some new c++20 feature.

It's not meant to be super performant nor very sophisticated by any mean :)

### Necessary Dependencies
To use auto-ekf you'll need to have:

* A C++ compiler that supports C++20. 
* [CMake 3.14 or higher](https://cmake.org/)
* [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [autodiff](https://github.com/autodiff/autodiff)

### Optional Dependencies
The examples make use of a C++ wrapper of matplotlib named [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) for visualization purposes.
This package should be fetched at configure time though.

### Tests
Tests have been set up using [Catch2](https://github.com/catchorg/Catch2) library. 
Anyway, no test has been written so far. 

auto-ekf is basically a wrapper around Eigen and autodiff which are both well tested so ... 

Fuzz testing could be useful but it does not seem to work with gcc which is the compiler I am currently using.

### Examples
The examples use the measurements data (lidar + radar) from Udacity Extended Kalman Filter repository that I have found online.

These data are used only for validation purposes; this work has nothing to do with the Udacity course about autonomous driving.

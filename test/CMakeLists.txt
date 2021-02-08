
# Testing library
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.9.1)
FetchContent_MakeAvailable(catch2)

add_library(catch_main STATIC catch_main.cc)
target_link_libraries(catch_main PUBLIC  Catch2::Catch2)

add_executable(tests tests.cc)
target_link_libraries(tests PRIVATE catch_main ekf)

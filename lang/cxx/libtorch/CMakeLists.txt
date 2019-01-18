cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(hw_libtorch)

find_package(Torch REQUIRED)

add_executable(hw_libtorch hw_libtorch.cxx)
target_link_libraries(hw_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET hw_libtorch PROPERTY CXX_STANDARD 11)

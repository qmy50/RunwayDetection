cmake_minimum_required(VERSION 3.12)
project(lane_cpp)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
pybind11_add_module(lane_cpp
    src/lane_postprocess.cpp
    src/lane_bind.cpp
)
target_include_directories(lane_cpp PRIVATE include)
target_compile_features(lane_cpp PRIVATE cxx_std_14)
install(TARGETS lane_cpp DESTINATION .)
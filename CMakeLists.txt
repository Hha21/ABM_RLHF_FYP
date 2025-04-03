cmake_minimum_required(VERSION 3.14)
project(cpp_env)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_env
    src_cpp/bindings.cpp
    src_cpp/Environment.cpp
    src_cpp/Firm.cpp
    src_cpp/Sector.cpp
    src_cpp/Parameters.cpp
)

target_include_directories(cpp_env PRIVATE include_cpp)
target_compile_features(cpp_env PRIVATE cxx_std_17)
cmake_minimum_required(VERSION 3.13.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

project(
    PythonCDT
    VERSION 0.1.0
    DESCRIPTION "Software surface data rasterizer library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(CDT/CDT CDT)
add_subdirectory(pybind11)
pybind11_add_module(PythonCDT cdt_bindings.cpp)
target_include_directories(
        PythonCDT PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(PythonCDT PRIVATE CDT::CDT)
# Use rasterizer as pre-compiled header for faster test-only re-compiles
#target_precompile_headers(PythonCDT PRIVATE include/rasterizer.h)

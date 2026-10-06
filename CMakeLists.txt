cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(CDT/CDT)

pybind11_add_module(_core src/cdt_bindings.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_definitions(_core PRIVATE VERSION_INFO="${SKBUILD_PROJECT_VERSION_FULL}")
target_link_libraries(_core PRIVATE CDT::CDT)

# Installed as a component, so that CDT's own install rules stay out of the wheel
install(TARGETS _core DESTINATION pythoncdt COMPONENT python)

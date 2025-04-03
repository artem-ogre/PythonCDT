cmake_minimum_required(VERSION 3.18.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  CDT
  GIT_REPOSITORY https://github.com/artem-ogre/CDT.git
  GIT_TAG 4b4181713c73cf0a49a5b88fb3df4acca7436235
  SOURCE_SUBDIR CDT
)
FetchContent_MakeAvailable(CDT)

pybind11_add_module(${SKBUILD_PROJECT_NAME} MODULE cdt_bindings.cpp)
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE pybind11::module)
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE CDT::CDT)
install(TARGETS ${SKBUILD_PROJECT_NAME} LIBRARY DESTINATION .)

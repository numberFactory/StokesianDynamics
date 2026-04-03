cmake_minimum_required(VERSION 3.22)
project(lubrication LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED NO_MODULE)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_DIR
)
find_package(nanobind CONFIG REQUIRED)

set(SOURCES src/lubrication.cpp)
nanobind_add_module(lubrication ${SOURCES})

target_link_libraries(lubrication PRIVATE Eigen3::Eigen)

target_compile_options(lubrication PRIVATE
  -fPIC
  -Wall
  -O3
  -march=native
)

install(TARGETS lubrication
        DESTINATION ${Python_SITEARCH}/StokesianDynamics)

file(GLOB PYTHON_SOURCES src/*.py)
install(FILES ${PYTHON_SOURCES}
        DESTINATION ${Python_SITEARCH}/StokesianDynamics)
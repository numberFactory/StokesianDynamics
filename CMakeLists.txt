cmake_minimum_required(VERSION 3.22)
project(Lubrication_Class LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake/"
    "$ENV{CONDA_PREFIX}/share/cmake/Modules"
)

# Preprocessor definitions
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
if(DOUBLEPRECISION)
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()

# Pass the source file path as a macro so __FILENAME__ resolves correctly
# for loading the resistance coefficient files at runtime
add_compile_definitions(
  __FILENAME__="${CMAKE_CURRENT_SOURCE_DIR}/."
)

# Python + Eigen
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED NO_MODULE)

# Locate nanobind via the active Python environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_DIR
)
find_package(nanobind CONFIG REQUIRED)

# Build the nanobind extension module
set(SOURCES Lubrication_Class.cpp)
nanobind_add_module(Lubrication_Class ${SOURCES})

target_link_libraries(Lubrication_Class PRIVATE Eigen3::Eigen)

target_compile_options(Lubrication_Class PRIVATE
  -fPIC
  -Wall
  -O3               # enable full optimisation for the lubrication numerics
  -march=native     # tune for the build machine's CPU
)

# Install the extension and any Python helpers alongside it
install(TARGETS Lubrication_Class
        DESTINATION ${Python_SITEARCH}/Lubrication)

file(GLOB PYTHON_SOURCES *.py)
install(FILES ${PYTHON_SOURCES}
        DESTINATION ${Python_SITEARCH}/Lubrication)
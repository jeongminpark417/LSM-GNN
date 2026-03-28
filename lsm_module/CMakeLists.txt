# lsm_module — PyBind11 + CUDA extension (BAM NVMe feature store)
#
#   include/lsm_module/   API headers + device shims
#   src/                  gids_nvme.cu
#   python/BAM_Feature_Store/  templates → ${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store/

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(lsm_module
  VERSION 0.1.1
  DESCRIPTION "PyBind11 CUDA bindings for BAM NVMe feature store"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()

find_package(Python3 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.Embed)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE _lsm_python_platlib
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _lsm_platlib_rc)
if(NOT _lsm_platlib_rc EQUAL 0 OR _lsm_python_platlib STREQUAL "")
  message(FATAL_ERROR "Could not query Python platlib (rc=${_lsm_platlib_rc}).")
endif()

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cli_rc)
  if(_pybind11_cli_rc EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR
      "pybind11 not found. Try: ${Python3_EXECUTABLE} -m pip install \"pybind11[global]\"")
  endif()
endif()

find_package(OpenMP)

set(LSM_GNN_BAM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../bam" CACHE PATH
  "BAM tree (headers under include/, libnvm under build/)")

set(NVM_LIBRARY "" CACHE FILEPATH
  "Optional path to libnvm.so; if empty, search under LSM_GNN_BAM_ROOT")

if(NVM_LIBRARY)
  if(NOT EXISTS "${NVM_LIBRARY}")
    message(FATAL_ERROR "NVM_LIBRARY is not a file: \"${NVM_LIBRARY}\"")
  endif()
else()
  find_library(NVM_LIBRARY
    NAMES nvm libnvm.so
    PATHS
      "${LSM_GNN_BAM_ROOT}/build/lib"
      "${LSM_GNN_BAM_ROOT}/build/module"
    NO_DEFAULT_PATH)
  if(NOT NVM_LIBRARY)
    message(FATAL_ERROR
      "libnvm not found under \"${LSM_GNN_BAM_ROOT}\". "
      "Build BAM or set NVM_LIBRARY / LSM_GNN_BAM_ROOT.")
  endif()
endif()
mark_as_advanced(NVM_LIBRARY)

message(STATUS "lsm_module: BAM root  ${LSM_GNN_BAM_ROOT}")
message(STATUS "lsm_module: libnvm    ${NVM_LIBRARY}")

set(_lsm_py_pkg "${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store")

pybind11_add_module(BAM_Feature_Store MODULE src/gids_nvme.cu)

target_include_directories(BAM_Feature_Store
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${LSM_GNN_BAM_ROOT}/include"
    "${LSM_GNN_BAM_ROOT}/include/freestanding/include")

target_link_libraries(BAM_Feature_Store PRIVATE "${NVM_LIBRARY}")

if(OpenMP_CXX_FOUND)
  target_link_libraries(BAM_Feature_Store PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(BAM_Feature_Store PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

set_target_properties(BAM_Feature_Store PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${_lsm_py_pkg}")

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/python/BAM_Feature_Store/__init__.py"
  "${_lsm_py_pkg}/__init__.py"
  COPYONLY)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/python/BAM_Feature_Store/setup.py.in"
  "${_lsm_py_pkg}/setup.py"
  @ONLY)

install(TARGETS BAM_Feature_Store
  LIBRARY DESTINATION "${_lsm_python_platlib}/BAM_Feature_Store"
  COMPONENT python)
install(FILES "${_lsm_py_pkg}/__init__.py"
  DESTINATION "${_lsm_python_platlib}/BAM_Feature_Store"
  COMPONENT python)

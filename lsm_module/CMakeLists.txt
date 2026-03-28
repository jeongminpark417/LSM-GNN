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

# BAM atomics / PTX need sm_70+. Pin the module target so stale global arch (e.g. 52) cannot break ptxas.
set(LSM_CUDA_ARCHITECTURES "70;80" CACHE STRING
  "CUDA architectures for BAM_Feature_Store (minimum sm_70).")
foreach(_lsm_a IN LISTS LSM_CUDA_ARCHITECTURES)
  string(TOLOWER "${_lsm_a}" _lsm_al)
  if(_lsm_al STREQUAL "native" OR _lsm_al STREQUAL "all" OR _lsm_al STREQUAL "all-major")
    message(FATAL_ERROR
      "lsm_module: LSM_CUDA_ARCHITECTURES='${_lsm_a}' is not supported. Use e.g. -DLSM_CUDA_ARCHITECTURES=80")
  endif()
  if(_lsm_a MATCHES "^[0-9]+")
    string(REGEX REPLACE "^([0-9]+).*" "\\1" _lsm_maj "${_lsm_a}")
    if(_lsm_maj LESS 70)
      message(FATAL_ERROR
        "lsm_module: LSM_CUDA_ARCHITECTURES entry '${_lsm_a}' is below minimum sm_70.")
    endif()
  endif()
endforeach()

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

# Do not use find_library(NVM_LIBRARY ...) when NVM_LIBRARY is a cached empty
# string: CMake treats that as "already set" and skips the search. Use a
# scratch variable, then publish the result to NVM_LIBRARY.
set(NVM_LIBRARY "" CACHE FILEPATH
  "Optional path to libnvm.so; if empty, search under LSM_GNN_BAM_ROOT")

if(NVM_LIBRARY)
  if(NOT EXISTS "${NVM_LIBRARY}")
    message(FATAL_ERROR "NVM_LIBRARY is not a file: \"${NVM_LIBRARY}\"")
  endif()
else()
  find_library(_lsm_nvm_lib
    NAMES nvm
    PATHS
      "${LSM_GNN_BAM_ROOT}/build/lib"
      "${LSM_GNN_BAM_ROOT}/build/module"
    NO_DEFAULT_PATH)
  if(NOT _lsm_nvm_lib)
    message(FATAL_ERROR
      "libnvm not found under \"${LSM_GNN_BAM_ROOT}\".\n"
      "Build the BAM userspace library (produces build/lib/libnvm.so), for example:\n"
      "  cmake -S \"${LSM_GNN_BAM_ROOT}\" -B \"${LSM_GNN_BAM_ROOT}/build\"\n"
      "  cmake --build \"${LSM_GNN_BAM_ROOT}/build\" -j --target libnvm\n"
      "Then re-run cmake, or pass the library explicitly:\n"
      "  cmake .. -DNVM_LIBRARY=\"${LSM_GNN_BAM_ROOT}/build/lib/libnvm.so\"\n"
      "If bam lives elsewhere: -DLSM_GNN_BAM_ROOT=/path/to/bam")
  endif()
  set(NVM_LIBRARY "${_lsm_nvm_lib}" CACHE FILEPATH
    "Optional path to libnvm.so; if empty, search under LSM_GNN_BAM_ROOT" FORCE)
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
  LIBRARY_OUTPUT_DIRECTORY "${_lsm_py_pkg}"
  CUDA_ARCHITECTURES "${LSM_CUDA_ARCHITECTURES}")

message(STATUS "lsm_module: CUDA_ARCHITECTURES (target) ${LSM_CUDA_ARCHITECTURES}")

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

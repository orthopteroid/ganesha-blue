cmake_minimum_required(VERSION 3.9)

project(example1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_VERBOSE_MAKEFILE TRUE)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr") # cuda 8.0 needed for std::array operator[] inside __device__ code
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DDEBUG")
endif()

# sssh! requires hack to FindCUDA.cmake file:
# add 'OR ${file} MATCHES "\\.cpp$"' to line starting with 'if((${file} MATCHES "\\.cu$"'
cuda_add_executable(example1 example1.cpp)
target_include_directories(example1 PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES})

add_subdirectory(clionShadow) # clion index-friendly subproject

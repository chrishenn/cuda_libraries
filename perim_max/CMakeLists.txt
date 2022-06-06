cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Create project, specify library name
SET(PROJECT_NAME perim_max)
project(${PROJECT_NAME} LANGUAGES CUDA CXX)

# specify your path to libtorch
#list(APPEND CMAKE_PREFIX_PATH "/om2/user/chenn/libtorch/")
list(APPEND CMAKE_PREFIX_PATH "/home/chris/Documents/libtorch")
FIND_PACKAGE(Torch REQUIRED)

# CUDA PACKAGE
FIND_PACKAGE( CUDA REQUIRED )
message("CUDA FOUND")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER clang++)

# Source files
file(GLOB INTERNAL_SOURCE ${PROJECT_SOURCE_DIR}/*.cpp )
file(GLOB CUDA_SOURCE ${PROJECT_SOURCE_DIR}/*.cu)
file(GLOB INTERNAL_HEADERS ${PROJECT_SOURCE_DIR}/*.h )

# Define our library target
add_library(${PROJECT_NAME} SHARED ${INTERNAL_SOURCE} ${INTERNAL_HEADERS} ${CUDA_SOURCE} )

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# ==================================================================================================
# @brief Global compiler settings for the whole project.
# @note $ENV{CUDA_CC} must be the path to a C compiler which is compatible with NVCC.
# ==================================================================================================

# Generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

# If not gnu
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "[playground] Unsupported compiler")
endif()

# [Bug] 
# Using compile flag `-fopenmp` with CUDA could cause some errors for 
# clangd-18.1.3 code analysis. The error message is as follows:
#   No library 'libomptarget-nvptx-sm_86.bc' found in the default clang lib 
#   directory or in LIBRARY_PATH; use '--libomptarget-nvptx-bc-path' to specify 
#   nvptx bitcode libraryclang(drv_omp_offload_target_missingbcruntime)
# This error is not a problem for the compilation process, but annoying. 
# I guess it is a bug of clangd. So, I comment out the following line:
#
## add_compile_options(-fopenmp)

add_compile_options(-save-temps)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin $ENV{CUDA_CC}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -lineinfo")

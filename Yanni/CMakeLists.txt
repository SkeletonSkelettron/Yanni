cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(yanni LANGUAGES CXX CUDA)
include(CTest)
set(HEADER_FILES C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include/cuda.h D:/vcpkg/installed/x64-windows/include/json.hpp)

add_executable(yanni Yanni.cpp)

target_include_directories(yanni PRIVATE 'D:/vcpkg/installed/x64-windows/include')
target_include_directories(yanni PRIVATE 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include')   
target_compile_features(yanni PUBLIC cxx_std_11)
 
set_target_properties( yanni PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(yanni PROPERTIES CUDA_ARCHITECTURES "75") 




# cmake_minimum_required(VERSION 3.24)
# enable_language(CUDA)
# CUDA_ADD_LIBRARY(cudaYanniLib
# cudaClass.cu
# OPTIONS "-G -lineinfo"
# )

# set_target_properties(cudaYanniLib PROPERTIES CUDA_ARCHITECTURES "75")
# # set_target_properties(cudaYanniLib
# #                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_link_libraries(cudaYanniLib)                       
# # CUDA_ADD_LIBRARY(cudaYanniLib SHARED
# #     activationFunctionsCuda.cu
# #     enumsCuda.cuh
# #     layerCuda.cuh
# #     lossFunctionsCuda.cu
# #     neuNetCuda.cu
# #     # ramdom.cu
# #     # statisticFunctionsCuda.cuh
# # )
# # set_target_properties(cudaYanniLib PROPERTIES CUDA_ARCHITECTURES "75")
# # TARGET_LINK_LIBRARIES(cudaYanniLib ${CUDA_LIBRARIES})

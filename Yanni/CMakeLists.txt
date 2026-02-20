enable_language(CUDA)

CUDA_ADD_LIBRARY(cudaYanniLib
    cuda/activationFunctionsCuda.cu
    cuda/lossFunctionsCuda.cu
    cuda/neuNetCuda.cu
    cuda/random.cu
    OPTIONS "-G -lineinfo"
)

target_include_directories(cudaYanniLib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/data
)

set_target_properties(cudaYanniLib PROPERTIES CUDA_ARCHITECTURES "75")
# set_target_properties(cudaYanniLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cudaYanniLib ${CUDA_LIBRARIES})

enable_language(CUDA)
CUDA_ADD_LIBRARY(cudaYanniLib
cudaClass.cu
OPTIONS "-G -lineinfo"
)

set_target_properties(cudaYanniLib PROPERTIES CUDA_ARCHITECTURES "75")
# set_target_properties(cudaYanniLib
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(cudaYanniLib)                       
# CUDA_ADD_LIBRARY(cudaYanniLib SHARED
#     activationFunctionsCuda.cu
#     enumsCuda.cuh
#     layerCuda.cuh
#     lossFunctionsCuda.cu
#     neuNetCuda.cu
#     # ramdom.cu
#     # statisticFunctionsCuda.cuh
# )
# set_target_properties(cudaYanniLib PROPERTIES CUDA_ARCHITECTURES "75")
# TARGET_LINK_LIBRARIES(cudaYanniLib ${CUDA_LIBRARIES})

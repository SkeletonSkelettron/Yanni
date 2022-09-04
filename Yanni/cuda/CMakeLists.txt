# Make an explicit list of all source files in Yanni_INC. This is important
# because CMake is not a build system: it is a build system generator. Suppose
# you add a file foo.cpp to src/ after running cmake .. . If you set
# Yanni_INC with `file(GLOB ... )`, this is not passed to the makefile; it
# doesn't know that foo.cpp exists and will not re-run cmake. Your
# collaborator's builds will fail and it will be unclear why. Whether you use
# file(GLOB ...) or not, you will need to re-run cmake, but with an explicit
# file list, you know beforehand why your code isn't compiling. 


set(Yanni_CUDA_INC
    activationFunctionsCuda.cu
    activationFunctionsCuda.cuh
    enumsCuda.cuh
    gradientFunctionsCuda.cuh
    layerCuda.cuh
    learningRateFunctionsCuda.cuh
    lossFunctionsCuda.cu
    lossFunctionsCuda.cuh
    neuNetCuda.cu
    neuralNetworkCuda.cuh
    statisticFunctionsCuda.cuh
    )

# Form the full path to the source files...
PREPEND(Yanni_CUDA_INC)
# ... and pass the variable to the parent scope.
set(Yanni_CUDA_INC ${Yanni_CUDA_INC}  PARENT_SCOPE)
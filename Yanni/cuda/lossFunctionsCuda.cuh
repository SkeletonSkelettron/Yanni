#ifndef LOSSFUNCTIONSCUDA_H
#define LOSSFUNCTIONSCUDA_H
#include "cuda.h"
#include "cuda_runtime.h"
#include "../include/enums.h"

__device__ float  KullbackLeiblerDivergenceCuda(float* roHat, float& ro, int start, int end);

__device__ float  KullbackLeiblerDivergenceDerivativeCuda(float& output, float& target);

__device__ float  BinaryCrossentropyCuda(float* output, float* target, int targetSize);

__device__ float  BinaryCrossentropyDerivativeCuda(float& output, float& target, int size);

__device__ float  _CELCuda(float& output, float& target);

__device__ float CELCuda(float* output, float* target, int size);

__device__ float MSLCuda(float& output, float& target);

__device__ float MSLCuda(float* output, float* target, int start, int end, int outputSize);

__device__ float CELDerevativeCuda(float& output, float& target);

__device__ float CalculateLossFunctionCuda(int& function, float* output, float* target, int start, int end, int outputSize);

__device__ float DifferentiateLossWithCuda(float& output, float& target, int& function, int size);

#endif //LOSSFUNCTIONSCUDA_H

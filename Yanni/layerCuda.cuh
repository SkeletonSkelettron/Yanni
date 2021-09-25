#ifndef LAYERCUDA_H
#define LAYERCUDA_H
#include "enums.h"
#include "activationFunctionsCuda.cuh"
__device__ struct LayerCuda
{
	size_t Size;
	int BatchSize;
	int ActivationFunction;
	int LayerType;
	size_t IndexVectorSize;
	size_t IndexVectorForNextLayerSize;
	float DropOutSize;
	bool UsingBias;
	int WeightsSize;
	float* RoHat;
	size_t* IndexVector;
	size_t* IndexVectorForNextLayer;
	int* DropoutNeurons;
	float* Weights;
	float* TempWeights;
	float* Gradients;
	float* GradientsLR;
	float* Parameters;
	float* InputsBatch;
	float* GradientsBatch;
	float* OutputsBatch;
	float* TargetsBatch;

	__device__ inline float& GetNumberFromBatch(float* batch, size_t batchNumber, size_t count)
	{
		return batch[batchNumber * Size + count];
	}

	__device__ inline float& GetInputsBatch(size_t batchNumber, size_t count)
	{
		return InputsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetOutputsBatch(size_t batchNumber, size_t count)
	{
		return OutputsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetGradientsBatch(size_t batchNumber, size_t count)
	{
		return GradientsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetTargetsBatch(size_t batchNumber, size_t count)
	{
		return TargetsBatch[batchNumber * Size + count];
	}

	__device__ inline float* GetInputsBatch(size_t batchNumber)
	{
		return &InputsBatch[batchNumber * Size];
	}
	__device__ inline float* GetOutputsBatch(size_t batchNumber)
	{
		return &OutputsBatch[batchNumber * Size];
	}
	__device__ inline float* GetGradientsBatch(size_t batchNumber)
	{
		return &GradientsBatch[batchNumber * Size];
	}
	__device__ inline float* GetTargetsBatch(size_t batchNumber)
	{
		return &TargetsBatch[batchNumber * Size];
	}
	__host__ __device__ LayerCuda() {}

	//---------------------------------------------------
	__device__ void CalculateInputs(
		int prevLayerSize,
		float* prevLayerOutputBatch,
		size_t* prevLayerIndexes,
		size_t& prevLayerIndexVectorSize,
		bool training,
		size_t batch,
		size_t start,
		size_t end)
	{
		float result;
		int biasShift = UsingBias ? 1 : 0;
		int k = 0, i = 0, l = 0, w = 0;
		for (int kk = start; kk < end; kk++)
		{
			k = IndexVector[kk];
			result = 0.0;
			for (int ii = 0; ii < prevLayerIndexVectorSize; ii++)
			{
				i = prevLayerIndexes[ii];
				l = batch * prevLayerSize;
				w = (k - biasShift) * prevLayerSize;
				result += prevLayerOutputBatch[l + i] * Weights[w + i];
			}
			GetInputsBatch(batch, k) = result;
		}
	}

	//---------------------------------------------------
	__device__ void CalculateOutputs(size_t& batch, size_t start, size_t end, bool training, bool countingRohat)
	{
		//TODO ჩასამატებელია SoftMax რეალიზაცია 
		ActivateWithCuda(
			GetInputsBatch(batch),
			GetOutputsBatch(batch),
			IndexVector, start, end, ActivationFunction);
		if (UsingBias)
			GetOutputsBatch(batch, 0) = GetInputsBatch(batch, 0);
	}

	float getLayerSize()
	{
		return BatchSize * Size * sizeof(float) * 2 + WeightsSize * BatchSize * sizeof(float) + (LayerType == 2 ? BatchSize * Size * sizeof(float) * 2.0f : 0.0f);
	}
};

#endif //LAYERCUDA_H
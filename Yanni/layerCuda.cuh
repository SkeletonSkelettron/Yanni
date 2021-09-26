#ifndef LAYERCUDA_H
#define LAYERCUDA_H
#include "enums.h"
#include "activationFunctionsCuda.cuh"
__device__ struct LayerCuda
{
	int Size;
	int BatchSize;
	int ActivationFunction;
	int LayerType;
	int IndexVectorSize;
	int IndexVectorForNextLayerSize;
	float DropOutSize;
	bool UsingBias;
	int WeightsSize;
	float* RoHat;
	int* IndexVector;
	int* IndexVectorForNextLayer;
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

	__device__ inline float& GetNumberFromBatch(float* batch, int batchNumber, int count)
	{
		return batch[batchNumber * Size + count];
	}

	__device__ inline float& GetInputsBatch(int batchNumber, int count)
	{
		return InputsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetOutputsBatch(int batchNumber, int count)
	{
		return OutputsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetGradientsBatch(int batchNumber, int count)
	{
		return GradientsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetTargetsBatch(int batchNumber, int count)
	{
		return TargetsBatch[batchNumber * Size + count];
	}

	__device__ inline float* GetInputsBatch(int batchNumber)
	{
		return &InputsBatch[batchNumber * Size];
	}
	__device__ inline float* GetOutputsBatch(int batchNumber)
	{
		return &OutputsBatch[batchNumber * Size];
	}
	__device__ inline float* GetGradientsBatch(int batchNumber)
	{
		return &GradientsBatch[batchNumber * Size];
	}
	__device__ inline float* GetTargetsBatch(int batchNumber)
	{
		return &TargetsBatch[batchNumber * Size];
	}
	__host__ __device__ LayerCuda() {}

	//---------------------------------------------------
	__device__ void CalculateInputs(
		int prevLayerSize,
		float* prevLayerOutputBatch,
		int* prevLayerIndexes,
		int& prevLayerIndexVectorSize,
		bool training,
		int batch,
		int start,
		int end)
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
	__device__ void CalculateOutputs(int& batch, int start, int end, bool training, bool countingRohat)
	{
		//TODO ჩასამატებელია SoftMax რეალიზაცია 
		ActivateWithCuda(
			GetInputsBatch(batch),
			GetOutputsBatch(batch),
			IndexVector, start, end, ActivationFunction);
		if (UsingBias)
			GetOutputsBatch(batch, 0) = GetInputsBatch(batch, 0);
	}

};

#endif //LAYERCUDA_H
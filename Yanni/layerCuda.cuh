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
	float* GradientsBatch;
	float* Parameters;
	float* Inputs;
	float* Outputs;
	float* Targets;

	__device__ inline float& GetNumberFromBatch(float* batch, int batchNumber, int count)
	{
		return batch[batchNumber * Size + count];
	}

	__device__ inline float& GetInputsBatch(int batchNumber, int count)
	{
		return Inputs[batchNumber * Size + count];
	}
	__device__ inline float& GetOutputsBatch(int batchNumber, int count)
	{
		return Outputs[batchNumber * Size + count];
	}
	__device__ inline float& GetGradientsBatch(int batchNumber, int count)
	{
		return GradientsBatch[batchNumber * Size + count];
	}
	__device__ inline float& GetTargetsBatch(int batchNumber, int count)
	{
		return Targets[batchNumber * Size + count];
	}

	__device__ inline float* GetInputsBatch(int batchNumber)
	{
		return &Inputs[batchNumber * Size];
	}
	__device__ inline float* GetOutputsBatch(int batchNumber)
	{
		return &Outputs[batchNumber * Size];
	}
	__device__ inline float* GetGradientsBatch(int batchNumber)
	{
		return &GradientsBatch[batchNumber * Size];
	}
	__device__ inline float* GetTargetsBatch(int batchNumber)
	{
		return &Targets[batchNumber * Size];
	}
	__host__ __device__ LayerCuda() {}

	//---------------------------------------------------
	__device__ void CalculateInputs(
		int prevLayerSize,
		float* prevLayerOutputs,
		int* prevLayerIndexes,
		int& prevLayerIndexVectorSize,
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
				result += prevLayerOutputs[l + i] * Weights[w + i];
			}
			Inputs[batch * Size + k] = result;
		}
	}

	//---------------------------------------------------
	__device__ void CalculateOutputs(int& batch, int start, int end, bool training, bool countingRohat)
	{
		//TODO ჩასამატებელია SoftMax რეალიზაცია 
		ActivateWithCuda(
			&Inputs[batch * Size],
			&Outputs[batch * Size],
			IndexVector, start, end, ActivationFunction);
		if (UsingBias)
			Outputs[batch * Size] = Inputs[batch * Size];
	}

};

#endif //LAYERCUDA_H
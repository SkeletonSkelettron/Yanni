#include "lossFunctionsCuda.cuh"
#include <vector>
#include <math.h>
#include <cmath>


float  KullbackLeiblerDivergenceCuda(float* roHat, float& ro, size_t start, size_t end)
{
	float sum = 0.0;
	for (size_t i = start; i < end; i++)
		sum += ro * log(ro / roHat[i]) + (1 - ro) * log((1 - ro) / (1 - roHat[i]));
	return sum;
}

float  KullbackLeiblerDivergenceDerivativeCuda(float& output, float& target)
{
	//TODO არ მუშაობს
	return log(output / target) + 1 / target;
}

float  BinaryCrossentropyCuda(float* output, float* target, size_t targetSize)
{
	float sum = 0;
	for (size_t i = 0; i < targetSize; i++)
	{
		sum += target[i] * log(output[i]) - (1 - target[i]) * log(1 - output[i]);
	}
	return -sum / targetSize;
}

float  BinaryCrossentropyDerivativeCuda(float& output, float& target, size_t size)
{
	return (-target / output + (1 - target) / (1 - output)) / size;
}

float  _CELCuda(float& output, float& target)
{
	return -target * log(output) - (1 - target) * log(1 - output);
}
float CELCuda(float* output, float* target, size_t size)
{
	float sum = 0;
	for (size_t i = 0; i < size; i++)
	{
		sum += _CELCuda(output[i], target[i]);
	}
	return sum / size;
}
float MSLCuda(float& output, float& target)
{
	return pow((target - output), 2) / 2;
}

float MSLCuda(float* output, float* target, size_t start, size_t end, size_t outputSize)
{
	float Sum = 0;
	for (size_t i = start; i < end; i++)
	{
		Sum += MSLCuda(target[i], output[i]) / outputSize;
	}
	return Sum;
}

float CELDerevativeCuda(float& output, float& target)
{
	return -target / output + (1 - target) / (1 - output);
}

float CalculateLossFunctionCuda(int& function, float* output, float* target, size_t start, size_t end, size_t outputSize)
{
	switch (function)
	{
	case static_cast<int>(NeuralEnums::LossFunctionType::MeanSquaredError): return MSLCuda(output, target, start, end, outputSize);
	case static_cast<int>(NeuralEnums::LossFunctionType::BinaryCrossentropy): return BinaryCrossentropyCuda(output, target, outputSize);
		//case NeuralEnums::LossFunctionType::KullbackLeiblerDivergence: return KullbackLeiblerDivergence(output, target);
	default:
		break;
	}
}
float DifferentiateLossWithCuda(float& output, float& target, int& function, size_t size)
{
	switch (function)
	{
	case static_cast<int>(NeuralEnums::LossFunctionType::MeanSquaredError): return output - target;
	case static_cast<int>(NeuralEnums::LossFunctionType::BinaryCrossentropy): return BinaryCrossentropyDerivativeCuda(output, target, size);
	case static_cast<int>(NeuralEnums::LossFunctionType::KullbackLeiblerDivergence): return KullbackLeiblerDivergenceDerivativeCuda(output, target);
	default:
		break;
	}
}

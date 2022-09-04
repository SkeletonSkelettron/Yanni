#include "../include/lossFunctions.h"
#include "../include/enums.h"
#include <vector>
#include <math.h>
#include <cmath>



float  KullbackLeiblerDivergence(float* roHat, float& ro, int start, int end)
{
	float sum = 0.0;
	for (size_t i = start; i < end; i++)
		sum += ro * log(ro / roHat[i]) + (1 - ro) * log((1 - ro) / (1 - roHat[i]));
	return sum;
}

float  KullbackLeiblerDivergenceDerivative(float& output, float& target)
{
	//TODO არ მუშაობს
	return log(output / target) + 1 / target;
}

float  BinaryCrossentropy(float* output, float* target, int targetSize)
{
	float sum = 0;
	for (int i = 0; i < targetSize; i++)
	{
		sum += target[i] * log(output[i]) - (1 - target[i]) * log(1 - output[i]);
	}
	return -sum / targetSize;
}

float  BinaryCrossentropyDerivative(float& output, float& target, int size)
{
	return (-target / output + (1 - target) / (1 - output)) / size;
}

float  _CEL(float& output, float& target)
{
	return -target * log(output) - (1 - target) * log(1 - output);
}
float CEL(float* output, float* target, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += _CEL(output[i], target[i]);
	}
	return sum / size;
}
float MSL(float& output, float& target)
{
	return pow((target - output), 2) / 2;
}

float MSL(float* output, float* target, int start, int end, int outputSize)
{
	float Sum = 0;
	for (unsigned long int i = start; i < end; i++)
	{
		Sum += MSL(target[i], output[i]) / outputSize;
	}
	return Sum;
}

float CELDerevative(float& output, float& target)
{
	return -target / output + (1 - target) / (1 - output);
}

float CalculateLossFunction(NeuralEnums::LossFunctionType& function, float* output, float* target, int start, int end, int outputSize)
{
	switch (function)
	{
	case NeuralEnums::LossFunctionType::MeanSquaredError: return MSL(output, target, start, end, outputSize);
	case NeuralEnums::LossFunctionType::BinaryCrossentropy: return BinaryCrossentropy(output, target, outputSize);
		//case NeuralEnums::LossFunctionType::KullbackLeiblerDivergence: return KullbackLeiblerDivergence(output, target);
	default:
		break;
	}
}
float DifferentiateLossWith(float& output, float& target, NeuralEnums::LossFunctionType& function, int size)
{
	switch (function)
	{
	case NeuralEnums::LossFunctionType::MeanSquaredError: return output - target;
	case NeuralEnums::LossFunctionType::BinaryCrossentropy: return BinaryCrossentropyDerivative(output, target, size);
	case NeuralEnums::LossFunctionType::KullbackLeiblerDivergence: return KullbackLeiblerDivergenceDerivative(output, target);
	default:
		break;
	}
}

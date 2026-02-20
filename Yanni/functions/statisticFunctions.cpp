#include <math.h>
#include "statisticFunctions.h"
using namespace std;

float Mean(float* x, int size)
{
	float sum = 0;

	for (unsigned long int i = 0; i < size; i++)
		sum += x[i];

	return sum / size;
}


float Max(float* x, int xSize)
{
	float max = x[0];

	for (unsigned long int i = 0; i < xSize; i++)
		if (x[i] > max)
			max = x[i];

	return max;
}

float Min(float* x, int xSize)
{
	float min = x[0];

	for (unsigned long int i = 0; i < xSize; i++)
		if (x[i] < min)
			min = x[i];

	return min;
}

void NormalizeN(std::vector<float>& input, std::vector<int>& minMax)
{
	float min = input[0];
	float max = input[0];

	for (unsigned long int i = 0; i < input.size(); i++)
	{
		if (input[i] > max)
			max = input[i];
		if (input[i] < max)
			min = input[i];
	}
	minMax[0] = min;
	minMax[1] = max;

	float range = max - min;
	for (int i = 0; i < input.size(); i++)
		input[i] = (input[i] - min) / range;
}

void DeNormalizeN(std::vector<float>& input, std::vector<int>& minMax)
{
	float min = minMax[0];
	float max = minMax[1];


	float range = max - min;
	for (int i = 0; i < input.size(); i++)
		input[i] = (int)(input[i] * range + min);
}
void Compress(float* input, int inputSize, int* minMax)
{
	float min = input[0];
	float max = input[0];

	for (unsigned long int i = 0; i < inputSize; i++)
	{
		if (input[i] >= max)
			max = input[i];
		if (input[i] <= max)
			min = input[i];
	}
	minMax[0] = min;
	minMax[1] = max;

	float range = max - min;
	for (int i = 0; i < inputSize; i++)
		input[i] = input[i] / range;
}

void DeCompress(std::vector<float>& input, std::vector<int>& minMax)
{
	float min = minMax[0];
	float max = minMax[1];


	float range = max - min;
	for (int i = 0; i < input.size(); i++)
		input[i] = (int)(input[i] * range);
}
float VarianceOfPopulation(float* x, int size)
{
	float mean = Mean(x, size), sumSq = 0;

	for (unsigned long int i = 0; i < size; i++)
	{
		float delta = x[i] - mean;

		sumSq += delta * delta;
	}

	return sumSq / size;
}
float StandardDeviationOfPopulation(float* x, int size)
{
	return sqrt(VarianceOfPopulation(x, size));
}

void Standartize(float* dataset, int size)
{
	const float Epsilon = 0.000001;
	auto mean = Mean(dataset, size);
	auto StDev = StandardDeviationOfPopulation(dataset, size);

	for (unsigned long int i = 0; i < size; i++)
		dataset[i] = (dataset[i] - mean) / sqrt(StDev + Epsilon);
}

void StandartizeLinearContract(float* dataset, int datasetSize, int* minMax, float& start, float& end)
{
	float min = Min(dataset, datasetSize);
	float max = Max(dataset, datasetSize);
	minMax[0] = min;
	minMax[1] = max;
	float range = max - min;
	for (unsigned long int i = 0; i < datasetSize; i++)
	{
		//if (dataset[i] != 0)
		dataset[i] = (end - start) * (dataset[i] - min) / range + start;
	}
}

void DeStandartizeLinearContract(float* dataset, int datasetSize, int* minMax)
{
	float min = minMax[0];
	float max = minMax[1];
	auto range = max - min;

	for (unsigned long int i = 0; i < datasetSize; i++)
	{
		//if (dataset[i] != 0)
		dataset[i] = (int)(range * (dataset[i] + 1.0) / 2.0 + min);
	}
}

void Normalize(float* dataset, int datasetSize)
{
	const float Epsilon = 0.000001;
	float range = Max(dataset, datasetSize) - Min(dataset, datasetSize);
	for (unsigned long int i = 0; i < datasetSize; i++)
	{
		dataset[i] = dataset[i] / (range + Epsilon);
	}
}
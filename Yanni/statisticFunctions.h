#ifndef STATISTICFUNCTIONS_H
#define STATISTICFUNCTIONS_H

#include <vector>
#include <math.h>


float Mean(float* x, size_t size);

float Max(float* x, size_t xSize);

float Min(float* x, size_t xSize);

void NormalizeN(std::vector<float>& input, std::vector<int>& minMax);

void DeNormalizeN(std::vector<float>& input, std::vector<int>& minMax);

void Compress(float* input, size_t inputSize, int* minMax);

void DeCompress(std::vector<float>& input, std::vector<size_t>& minMax);

float VarianceOfPopulation(float* x, size_t size);

float StandardDeviationOfPopulation(float* x, size_t size);

void Standartize(float* dataset, size_t size);

void StandartizeLinearContract(float* dataset, size_t datasetSize, int* minMax, float& start, float& end);

void DeStandartizeLinearContract(float* dataset, size_t datasetSize, int* minMax);

void Normalize(float* dataset, size_t datasetSize);

#endif // STATISTICFUNCTIONS_H

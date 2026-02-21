#ifndef STATISTICFUNCTIONS_H
#define STATISTICFUNCTIONS_H

#include <vector>
#include <math.h>


float Mean(float* x, int size);

float Max(float* x, int xSize);

float Min(float* x, int xSize);

void NormalizeN(std::vector<float>& input, std::vector<int>& minMax);

void DeNormalizeN(std::vector<float>& input, std::vector<int>& minMax);

void Compress(float* input, int inputSize, int* minMax);

void DeCompress(std::vector<float>& input, std::vector<int>& minMax);

float VarianceOfPopulation(float* x, int size);

float StandardDeviationOfPopulation(float* x, int size);

void Standartize(float* dataset, int size);

void StandartizeLinearContract(float* dataset, int datasetSize, int* minMax, float& start, float& end);

void DeStandartizeLinearContract(float* dataset, int datasetSize, int* minMax);

void Normalize(float* dataset, int datasetSize);

#endif // STATISTICFUNCTIONS_H

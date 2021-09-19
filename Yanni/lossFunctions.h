#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

#include "enums.h"
#include <math.h>
#include <cmath>

float  KullbackLeiblerDivergence(float* roHat, float& ro, size_t start, size_t end);

float  KullbackLeiblerDivergenceDerivative(float& output, float& target);

float  BinaryCrossentropy(float* output, float* target, size_t targetSize);

float  BinaryCrossentropyDerivative(float& output, float& target, size_t size);

float  _CEL(float& output, float& target);

float CEL(float* output, float* target, size_t size);

float MSL(float& output, float& target);

float MSL(float* output, float* target, size_t start, size_t end, size_t outputSize);

float CELDerevative(float& output, float& target);
 
float CalculateLossFunction(NeuralEnums::LossFunctionType& function, float* output, float* target, size_t start, size_t end, size_t outputSize);

float DifferentiateLossWith(float& output, float& target, NeuralEnums::LossFunctionType& function, size_t size);

#endif //LOSSFUNCTIONS_H
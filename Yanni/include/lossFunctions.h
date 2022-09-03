#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

#include "enums.h"
#include <math.h>
#include <cmath>

float  KullbackLeiblerDivergence(float* roHat, float& ro, int start, int end);

float  KullbackLeiblerDivergenceDerivative(float& output, float& target);

float  BinaryCrossentropy(float* output, float* target, int targetSize);

float  BinaryCrossentropyDerivative(float& output, float& target, int size);

float  _CEL(float& output, float& target);

float CEL(float* output, float* target, int size);

float MSL(float& output, float& target);

float MSL(float* output, float* target, int start, int end, int outputSize);

float CELDerevative(float& output, float& target);

float CalculateLossFunction(NeuralEnums::LossFunctionType& function, float* output, float* target, int start, int end, int outputSize);

float DifferentiateLossWith(float& output, float& target, NeuralEnums::LossFunctionType& function, int size);

#endif //LOSSFUNCTIONS_H
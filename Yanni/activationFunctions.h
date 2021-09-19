#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include <vector>
#include <math.h>
#include <cmath>
#include "enums.h"
#include "statisticfunctions.h" 


//void BalanceWith(std::vector <float>& dataset, NeuralEnums::BalanceType BalancingMethod)
//{
//	switch (BalancingMethod)
//	{
//	case NeuralEnums::BalanceType::None: break;
//	case NeuralEnums::BalanceType::GaussianStandartization:
//	{
//		Standartize(dataset, dataset.size());
//		break;
//	}
//	case NeuralEnums::BalanceType::Normalization:
//	{
//		std::vector<int> tmp;
//		tmp.resize(2);
//		Compress(dataset,dataset.size() , tmp);
//		break;
//	}
//
//	default:
//		break;
//	}
//}

void ActivateWith(
	float* inputs,
	float* outputs,
	size_t* indexVector,
	size_t& start,
	size_t& end,
	NeuralEnums::ActivationFunction& function);

inline float SoftSign(float& x);

inline float SoftSignDerivative(float& x);

float SoftPlus(float& x);

inline float SoftPlusDerivative(float& x);

inline float SoftMax(float& x, float* layerInputs, size_t* indexVector, size_t& indexVectorSize);

inline float SoftMaxDerivative(float& x, float* inputs, size_t* indexVector, size_t& indexVectorSize);

inline float Sigmoid(float& x);

inline float SigmoidDerivative(float& x);

inline float ReLU(float& x);

inline float ReLUDerivative(float& x);

inline float Tanh(float& x);

inline float TanhDerivative(float& x);

inline float MReLU(float& x);

inline float MReLUDerivative(float& x);

inline float GeLU(float& x);

inline float GeLUDerivative(float& x);

int GetMaxIndex(float* outPut, size_t outpSize);

float exp1024(float x);

inline void GeLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void Sigmoid_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void Tanh_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void MReLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void ReLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void SoftMax_v(float* inputs, float* inputsSoftMax, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void SoftPlus_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void SoftSign_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

inline void Assign_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

float DifferentiateWith(float& x, NeuralEnums::ActivationFunction& function, float* inputs, bool* dropouts);

#endif ACTIVATIONFUNCTIONS_H
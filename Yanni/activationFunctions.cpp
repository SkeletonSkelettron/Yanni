#include "activationFunctions.h"
#include <vector>
#include <math.h>
#include <cmath>
#include "enums.h"
#include "statisticfunctions.h" 
#include <stdexcept>

const float PI2 = 6.283185307179586476f;
const float SQ2 = 1.414213562373095048f;

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
	NeuralEnums::ActivationFunction& function)
{
	switch (function)
	{
	case (NeuralEnums::ActivationFunction::Sigmoid):
	{
		Sigmoid_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::ReLU):
	{
		ReLU_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::MReLU):
	{
		MReLU_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::Tanh):
	{
		Tanh_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::GeLU):
	{
		GeLU_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::SoftPlus):
	{
		SoftPlus_v(inputs, outputs, indexVector, start, end);
		break;
	}
	case(NeuralEnums::ActivationFunction::SoftSign):
	{
		SoftSign_v(inputs, outputs, indexVector, start, end);
		break;
	}
	default:
		throw std::runtime_error("ActivationFunction not assigned");
	}
}

inline void GeLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = GeLU(inputs[indexVector[i]]);
	}
}
inline void Sigmoid_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = Sigmoid(inputs[indexVector[i]]);
	}
}
inline void Tanh_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = tanh(inputs[indexVector[i]]);
	}
}
inline void MReLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = MReLU(inputs[indexVector[i]]);
	}
}
inline void ReLU_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = ReLU(inputs[indexVector[i]]);
	}
}
inline void SoftMax_v(float* inputs, float* inputsSoftMax, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	//TODO მაინც კაი სანახავია როგორ მუშაობს
	//for (int i = 0; i < indexVectorSize; i++);
	// outputs[indexVectorSize[i]] = SoftMax(inputs[indexVectorSize[i]], inputsSoftMax, dropoutNeurons);
}
inline void SoftPlus_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = SoftPlus(inputs[indexVector[i]]);
	}
}
inline void SoftSign_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = SoftSign(inputs[indexVector[i]]);
	}
}

inline void Assign_v(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end)
{
	for (int i = start; i < end; i++)
	{
		outputs[indexVector[i]] = inputs[indexVector[i]];
	}
}

inline float SoftSign(float& x)
{
	return x / (abs(x) + 1);
}
inline float SoftSignDerivative(float& x)
{
	return  1.0f / pow(1.0f + abs(x), 2);
}

float SoftPlus(float& x)
{
	return log(1.0f + exp(x));
}
inline float SoftPlusDerivative(float& x)
{
	return  1.0f / (1.0f + exp(-x));
}
inline float SoftMax(float& x, float* layerInputs, size_t* indexVector, size_t& indexVectorSize)
{
	float sum = 0.0f;
	for (int i = 0; i < indexVectorSize; i++)
	{
		sum += exp(layerInputs[indexVector[i]]);
	}
	return exp(x) / sum;
}

inline float SoftMaxDerivative(float& x, float* inputs, size_t* indexVector, size_t& indexVectorSize)
{
	float y = SoftMax(x, inputs, indexVector, indexVectorSize);
	return y * (1.0f - y);
}

inline float Sigmoid(float& x)
{
	return  1.0f / (1.0f + exp(-x));
}

inline float SigmoidDerivative(float& x)
{
	float sigm = Sigmoid(x);
	return sigm * (1.0f - sigm);
}

inline float ReLU(float& x)
{
	return x <= 0.0f ? 0.0f : x;
}

inline float ReLUDerivative(float& x)
{
	return x == 0.0f ? 0.0f : 1.0f;
}

inline float Tanh(float& x)
{
	return tanh(x);
}

inline float TanhDerivative(float& x)
{
	return 1.0f - tanh(x) * tanh(x);
}

inline float MReLU(float& x)
{
	return x < 0.0f ? 0.0005f * x : x;
}

inline float MReLUDerivative(float& x)
{
	return x < 0.0f ? 0.0005f : 1.0;
}

inline float GeLU(float& x)
{
	return 0.5f * x * (1.0f + erf(x / SQ2));
}

inline float GeLUDerivative(float& x)
{
	return 0.5f + 0.5f * erf(x / SQ2) + x / (exp(-(x * x) / 2.0f) * pow(PI2, 0.5f));
}


int GetMaxIndex(float* outPut, size_t outpSize)
{
	int index = 0;
	float val = outPut[0];
	for (size_t i = 0; i < outpSize; i++)
	{
		if (outPut[i] > val)
		{
			val = outPut[i];
			index = i;
		}
	}
	return index;
}
//


float exp1024(float x)
{
	x = 1.0f + x / 256.0f;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}


float DifferentiateWith(float& x, NeuralEnums::ActivationFunction& function, float* inputs, bool* dropouts)
{
	switch (function)
	{
	case(NeuralEnums::ActivationFunction::Sigmoid):
	{
		return SigmoidDerivative(x);
		break;
	}
	case(NeuralEnums::ActivationFunction::ReLU):
	{
		return ReLUDerivative(x);
		break;
	}
	case(NeuralEnums::ActivationFunction::MReLU):
	{
		return MReLUDerivative(x);
		break;
	}
	case(NeuralEnums::ActivationFunction::Tanh):
	{
		return TanhDerivative(x);
		break;
	}
	case(NeuralEnums::ActivationFunction::GeLU):
	{
		return GeLUDerivative(x);
		break;
	}
	case(NeuralEnums::ActivationFunction::SoftPlus):
	{
		return SoftPlusDerivative(x);
		break;
	}
	default:
	{
		throw std::runtime_error("ActivationFunction not assigned");
		break;
	}
	}

}
#include "ActivationFunctionsCuda.cuh"
#include <vector>
#include <math.h>
#include <cmath>
#include "enums.h"
//#include "statisticfunctions.h" 
#include <stdexcept>
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


void ActivateWithCuda(
	float* inputs,
	float* outputs,
	int* indexVector,
	int& indexVectorSize,
	bool batch,
	int* drops,
	int& start,
	int& end,
	bool usingdrops,
	int& function)
{
	if (function == 1)
	{
		SigmoidCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 3)
	{
		ReLUCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 4)
	{
		MReLUCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 2)
	{
		TanhCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 6)
	{
		GeLUCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 7)
	{
		SoftPlusCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	else if (function == 8)
	{
		SoftSignCuda(inputs, outputs, indexVector, indexVectorSize, batch, drops, start, end, usingdrops);
		return;
	}
	//else
	//	throw std::runtime_error("ActivationFunction not assigned");
}


inline float SoftSignCuda(float& x)
{
	return x / (abs(x) + 1);
}
inline float SoftSignDerivativeCuda(float& x)
{
	return  1.0 / pow(1.0 + abs(x), 2);
}

inline float SoftPlusCuda(float& x)
{
	return log(1.0 + exp(x));
}
inline float SoftPlusDerivativeCuda(float& x)
{
	return  1.0 / (1.0 + exp(-x));
}
inline float SoftMaxCuda(float& x, float* layerInputs, int* indexVector, int& indexVectorSize)
{
	float sum = 0.0;
	for (int i = 0; i < indexVectorSize; i++)
	{
		sum += exp(layerInputs[indexVector[i]]);
	}
	return exp(x) / sum;
}

inline float SoftMaxDerivativeCuda(float& x, float* inputs, int* indexVector, int& indexVectorSize)
{
	float y = SoftMaxCuda(x, inputs, indexVector, indexVectorSize);
	return y * (1.0 - y);
}

inline float SigmoidCuda(float& x)
{
	return  1.0 / (1.0 + exp(-x));
}

inline float SigmoidDerivativeCuda(float& x)
{
	float sigm = SigmoidCuda(x);
	return sigm * (1.0 - sigm);
}

inline float ReLUCuda(float& x)
{
	return x <= 0.0 ? 0.0 : x;
}

inline float ReLUDerivativeCuda(float& x)
{
	return x == 0.0 ? 0.0 : 1.0;
}

inline float TanhCuda(float& x)
{
	return tanh(x);
}

inline float TanhDerivativeCuda(float& x)
{
	return 1.0 - tanh(x) * tanh(x);
}

inline float MReLUCuda(float& x)
{
	return x < 0.0 ? 0.0005 * x : x;
}

inline float MReLUDerivativeCuda(float& x)
{
	return x < 0.0 ? 0.0005 : 1.0;
}

inline float GeLUCuda(float& x)
{
	return 0.5 * x * (1.0 + erf(x / 1.414213562373095048));
}

inline float GeLUDerivativeCuda(float& x)
{
	return 0.5 + 0.5 * erf(x / 1.414213562373095048) + x / (exp(-(x * x) / 2.0) * pow(6.283185307179586476, 0.5));
}


int GetMaxIndexCuda(float* outPut, int outpSize)
{
	int index = 0;
	float val = outPut[0];
	for (unsigned long int i = 0; i < outpSize; i++)
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


float exp1024Cuda(float x)
{
	x = 1.0 + x / 256.0;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}


void GeLUCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = GeLUCuda(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = GeLUCuda(inputs[i]);
		}
	}
}
void SigmoidCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		int i;
		for (int ii = 0; ii < indexVectorSize; ii++)
		{
			i = indexVector[ii];
			outputs[i] = SigmoidCuda(inputs[i]);
		}
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = SigmoidCuda(inputs[i]);
		}
	}
}
void TanhCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = tanh(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = tanh(inputs[i]);
		}
	}
}
void MReLUCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = MReLUCuda(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = MReLUCuda(inputs[i]);
		}
	}
}
void ReLUCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = ReLUCuda(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = ReLUCuda(inputs[i]);
		}
	}
}
void SoftMaxCuda(float* inputs, float* inputsSoftMax, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	//TODO მაინც კაი სანახავია როგორ მუშაობს
	for (int i = 0; i < indexVectorSize; i++);
	// outputs[indexVectorSize[i]] = SoftMax(inputs[indexVectorSize[i]], inputsSoftMax, dropoutNeurons);
}
void SoftPlusCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = SoftPlusCuda(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = SoftPlusCuda(inputs[i]);
		}
	}
}
void SoftSignCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = SoftSignCuda(inputs[indexVector[i]]);
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = SoftSignCuda(inputs[i]);
		}
	}
}

void AssignCuda(float* inputs, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops)
{
	if (batch)
	{
		for (int i = 0; i < indexVectorSize; i++)
			outputs[indexVector[i]] = inputs[indexVector[i]];
	}
	else
	{
		for (int i = start; i < end; i++)
		{
			if (usingdrops && drops[i] == 1)
				continue;
			outputs[i] = inputs[i];
		}
	}
}

float DifferentiateWithCuda(float& x, int& function, float* inputs, int* dropouts)
{
	if (function == static_cast<int>(NeuralEnums::ActivationFunction::Sigmoid))
		return SigmoidDerivativeCuda(x);
	else if (function == static_cast<int>(NeuralEnums::ActivationFunction::ReLU))
		return ReLUDerivativeCuda(x);
	else if (function == static_cast<int>(NeuralEnums::ActivationFunction::MReLU))
		return MReLUDerivativeCuda(x);
	else if (function == static_cast<int>(NeuralEnums::ActivationFunction::Tanh))
		return TanhDerivativeCuda(x);
	else if (function == static_cast<int>(NeuralEnums::ActivationFunction::GeLU))
		return GeLUDerivativeCuda(x);
	else if (function == static_cast<int>(NeuralEnums::ActivationFunction::SoftPlus))
		return SoftPlusDerivativeCuda(x);
	//else
	//	throw std::runtime_error("ActivationFunction not assigned");
}
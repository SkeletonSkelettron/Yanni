#ifndef ACTIVATIONFUNCTIONSCUDA_H
#define ACTIVATIONFUNCTIONSCUDA_H
#include <cmath>
#include "enums.h"
#include "cuda.h"
#include "cuda_runtime.h"
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


__device__	void ActivateWithCuda(
	float* inputs,
	float* outputs,
	size_t* indexVector,
	size_t& start,
	size_t& end,
	int& function);

__device__	inline float SoftSignCuda(float& x);

__device__	inline float SoftSignDerivativeCuda(float& x);

__device__		float SoftPlusCuda(float& x);

__device__	inline float SoftPlusDerivativeCuda(float& x);

//__device__	inline float SoftMaxCuda(float& x, float* layerInputs, size_t* indexVector, size_t& indexVectorSize);

// __device__	inline float SoftMaxDerivativeCuda(float& x, float* inputs, int* indexVector, int& indexVectorSize);

__device__	inline float SigmoidCuda(float& x);

__device__	inline float SigmoidDerivativeCuda(float& x);

__device__	inline float ReLUCuda(float& x);

__device__	inline float ReLUDerivativeCuda(float& x);

__device__	inline float TanhCuda(float& x);

__device__	inline float TanhDerivativeCuda(float& x);

__device__	inline float MReLUCuda(float& x);

__device__	inline float MReLUDerivativeCuda(float& x);

__device__	inline float GeLUCuda(float& x);

__device__	inline float GeLUDerivativeCuda(float& x);

__device__	int GetMaxIndexCuda(float* outPut, int outpSize);

__device__	float exp1024Cuda(float x);

__device__	void GeLUCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void SigmoidCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void TanhCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void MReLUCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void ReLUCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

//__device__	void SoftMaxCuda(float* inputs, float* inputsSoftMax, float* outputs, int* indexVector, int& indexVectorSize, bool batch, int* drops, int& start, int& end, bool& usingdrops);

__device__	void SoftPlusCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void SoftSignCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	void AssignCuda(float* inputs, float* outputs, size_t* indexVector, size_t& start, size_t& end);

__device__	float DifferentiateWithCuda(float& x, int& function, float* inputs, int* dropouts);


#endif //ACTIVATIONFUNCTIONSCUDA_H
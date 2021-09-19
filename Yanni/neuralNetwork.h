#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <random>
#include <math.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <thread>
#include <condition_variable>
#include "enums.h"
#include "ActivationFunctions.h"
#include "WorkerThread.h"
#include "Layer.h"
#include "LearningRateFunctions.h"
#include "GradientFunctions.h"
#include "LossFunctions.h"
//https://stackoverflow.com/questions/16350473/why-do-i-need-stdcondition-variable


class NeuralNetwork
{
public:
	size_t ThreadCount;
	float LearningRate;
	std::vector<WorkerThread*> workers;
	NeuralEnums::NetworkType Type;
	NeuralEnums::LearningRateType LearningRateType;
	NeuralEnums::BalanceType BalanceType;
	NeuralEnums::LossFunctionType LossFunctionType;
	NeuralEnums::GradientType GradientType;
	NeuralEnums::Metrics Metrics;
	NeuralEnums::AutoEncoderType AutoEncoderType;
	NeuralEnums::LossCalculation LossCalculation;
	NeuralEnums::LogLoss LogLoss;
	int BatchSize;
	bool Cuda;
	Layer* Layers;
	size_t LayersSize;
	float*** GradientsTemp;
	float* lossesTmp;
	float ro;
	int iterations;
	float beta1Pow;
	float beta2Pow;
	float betaAELR;

	const float momentum = 0.9f;
	const float epsilon = 0.0000001f;
	const float startingLearningRate = 0.001f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;

	//weight decay parameter for sparce autoencoder
	float lambda = 0.7f;
	void NeuralNetworkInit();
	void InitializeWeights();
	void PrepareForTesting();
	float PropagateForwardThreaded(bool training, bool countingRohat);
	void PropagateBackThreaded();
	void PropagateBackDelegate(int i, size_t start, size_t end);
	void PropagateBackDelegateNew(int i, size_t start, size_t end);
	void PropagateBackDelegateBatch(size_t start, size_t end, int threadNum);
	void ShuffleDropoutsPlain();
	void CalculateWeightsBatch();
	void CalculateWeightsBatchSub(int i, size_t* prevLayerIndex, size_t prevLayerIndexSize, size_t start, size_t end, bool prevLayerUsingbias);
	float CalculateLoss(bool& training);
	void CalculateLossSub(size_t start, size_t end, size_t klbstart, size_t  klbend, float& loss);
	void CalculateLossBatchSub(size_t start, size_t end, float& loss);


	float GetLearningRateMultipliedByGrad(float& gradient, int& iterator, size_t& j);
	float Adam(float& gradient, size_t& j, int& iterator);
	float AdaGrad(float* gradients, float& gradient, size_t& j);
	float AdaDelta(float* gradients, float* parameters, float& Gradient, size_t& j);
	float AdamMod(float& Gradient, size_t& j, int& iterator);
	float AdaMax(float& gradient, size_t& j, int& iterator);
	float RMSProp(float* gradients, float& gradient, size_t& j);
};
#endif //NEURALNETWORK_H


/*#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>


// Forward declare the function in the .cu file
template <typename T>
std::vector<T> vectorActivationCUDA(std::vector<T>& inputs);

template <typename T> std::vector<T> cudaCalculate(std::vector<T>& output, std::vector<T>& weightsTensor)
{
	int cols = output.size();
	int rows = weightsTensor.size() / output.size();
	
	std::vector<double> p;
	p.resize(rows);

	printf("\nStarting CUDA computation...");
	///double startTime = timenow();

	// device pointers
	double* d_weightsTensor, * d_output, * d_result;

	cudaMalloc((void**)&d_weightsTensor, cols * rows * sizeof(T));
	cudaMalloc((void**)&d_result, cols * sizeof(T));
	cudaMalloc((void**)&d_output, cols * sizeof(T));
	// might need to flatten A...
	auto res0 = cublasSetVector(cols, sizeof(double), &(output[0]), 1, d_output, 1);
	//cudaMemcpy(d_x0, &x0, N*sizeof(float), cudaMemcpyHostToDevice);
	auto res1 = cublasSetMatrix(rows, cols, sizeof(double), &(weightsTensor[0]), rows, d_weightsTensor, rows);
	//cudaCheckErrors("cuda memcpy of A or x0 fail");

	double* temp;
	temp = (double*)malloc(cols * sizeof(temp));

	cublasHandle_t handle;
	cublasCreate(&handle);

	double alpha = 1.0L;
	double beta = 0.0L;
	auto res2 = cublasDgemv(handle, CUBLAS_OP_T, cols, rows , &alpha, d_weightsTensor, cols, d_output, 1, &beta, d_result, 1);

	// cublasGetVector(N, sizeof(float), d_temp, 1, temp, 1);
	cudaMemcpy(p.data(), d_result, rows * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	cudaFree(d_weightsTensor);
	cudaFree(d_output);
	return p;

}

template <typename T> std::vector<T> cudaActivate(std::vector<T>& inputs)
{
	std::vector<T> res = vectorActivationCUDA(inputs);
	return res;
}*/
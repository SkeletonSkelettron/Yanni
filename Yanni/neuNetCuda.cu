#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include "Enums.h"
#include "NeuralNetworkCuda.cuh"
#include "NeuralNetwork.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "MnistData.h"
#include <iostream>
#include <sstream>

#define ERRCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true, bool wait = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (wait) getch();
		if (abort) exit(code);
	}
}

__device__ int* results;
__device__ __constant__ NeuralNetworkCuda* neuNet;
__device__ LayerCuda* layersCd;
__device__ float* trainingSetCuda;
__device__ float* trainingLabelCuda;
__device__ float* testSetCuda;
__device__ float* testLabelCuda;


__managed__ float passAccuracy = 0.0;

__global__ void checkResults(int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < maxinputs)
		results[idx] = GetMaxIndexCuda(neuNet->Layers[neuNet->LayersSize - 1].GetOutputsBatch(idx), neuNet->Layers[neuNet->LayersSize - 1].Size)
		== GetMaxIndexCuda(&testLabelCuda[idx * neuNet->Layers[neuNet->LayersSize - 1].Size], neuNet->Layers[neuNet->LayersSize - 1].Size);
}
__global__ void clearResult(int size)
{
	for (size_t i = 0; i < size; i++)
	{
		results[i] = 0;
	}
}
__global__ void countAccuracyPercent()
{
	int counter = 0;
	for (int i = neuNet->MaxBachSize; i--;)
	{
		if (results[i])
			counter++;
	}
	passAccuracy = (float)counter / (float)neuNet->MaxBachSize;
}
__global__ void setBatchSize(int batchSize)
{
	neuNet->BatchSize = batchSize;
}

__global__ void checkLayers()
{
	auto t = neuNet->Layers;
	float fortmat = 0.0f;
	int k;
	//for (size_t i = 0; i < neuNet->Layers[3].Size; i++)
	//{
	//	fortmat = neuNet->Layers[3].Outputs[i];
	//	k = 0;
	//}
	// 	//	1.00021708	
	//	0.600195408	
	//	0.700086832	
	//	- 0.399848044	
	//	0.999847174	
	//	- 0.800137520	
	//	0.399938881	
	//	0.0998930335	
	//	0.999557257	
	//	0.229601517	
	//	0.169822901	
	//	0.159690067	
	//passAccuracy = k;
}
__global__ void copyLayers(int maxBatchSize, int* results_)
{
	neuNet->Layers = layersCd;
	neuNet->MaxBachSize = maxBatchSize;
	results = results_;
}

__global__ void initLayer(
	int layerIdx,
	float* weights,
	float* tempWeights,
	float* gradients,
	float* gradientslr,
	float* parameters,
	int* idxVec,
	int* idxNVec,
	int* neur,
	float* inputsBatch,
	float* outputsBatch,
	float* gradientsBatch,
	float* targetsBatch, int layerType)
{
	neuNet->Layers[layerIdx].Inputs = inputsBatch;
	neuNet->Layers[layerIdx].Outputs = outputsBatch;
	neuNet->Layers[layerIdx].Weights = weights;
	neuNet->Layers[layerIdx].TempWeights = tempWeights;
	neuNet->Layers[layerIdx].Gradients = gradients;
	neuNet->Layers[layerIdx].GradientsLR = gradientslr;
	neuNet->Layers[layerIdx].Parameters = parameters;
	neuNet->Layers[layerIdx].IndexVector = idxVec;
	neuNet->Layers[layerIdx].IndexVectorForNextLayer = idxNVec;
	neuNet->Layers[layerIdx].DropoutNeurons = neur;
	if (layerType == 2)
		neuNet->Layers[layerIdx].Targets = targetsBatch;

}


// this algorithm is valid if length of input layer <= length of target layer. if lt > li other implementation needed 
__global__ void assignData(int iter, int maxinputs, bool testing)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[0].Size;
	int j = idx % neuNet->Layers[0].Size;
	int l = idx % neuNet->Layers[neuNet->LayersSize - 1].Size;

	if (idx < maxinputs)
	{
		if (!testing)
		{
			neuNet->Layers[0].Outputs[idx] = trainingSetCuda[idx];

			if (idx < neuNet->BatchSize * neuNet->Layers[neuNet->LayersSize - 1].Size)
				neuNet->Layers[neuNet->LayersSize - 1].Targets[idx] = trainingLabelCuda[idx];
		}
		else
		{
			neuNet->Layers[0].Outputs[idx] = testSetCuda[idx];

			if (idx < neuNet->BatchSize * neuNet->Layers[neuNet->LayersSize - 1].Size)
				neuNet->Layers[neuNet->LayersSize - 1].GetTargetsBatch(batch, l) = testLabelCuda[idx];
		}
	}
}

__global__ void propagateForward(int k, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[k].IndexVectorSize;
	int j = idx % neuNet->Layers[k].IndexVectorSize;
	if (idx < maxinputs)
	{
		neuNet->Layers[k].CalculateInputs(neuNet->Layers[k - 1].Size, neuNet->Layers[k - 1].Outputs, neuNet->Layers[k - 1].IndexVectorForNextLayer, neuNet->Layers[k - 1].IndexVectorForNextLayerSize, batch, j, j + 1);
		neuNet->Layers[k].CalculateOutputs(batch, j, j + 1);
	}
}

__global__ void backProp(int k, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[k].IndexVectorSize;
	int j = idx % neuNet->Layers[k].IndexVectorSize;
	if (idx < maxinputs)
	{
		neuNet->PropagateBack(k, batch, j, j + 1);
		//neuNet->PropagateBack(k, batch, 0, maxinputs);
	}
}

__global__ void backPropCalcGradients(int layerIdx, int batchCount, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < maxinputs)
	{
		neuNet->PropagateBackCalculateGradients(layerIdx, batchCount, idx, idx + 1);
		//neuNet->PropagateBackCalculateGradients(layerIdx, batchCount, 0, maxinputs);
	}
}


void copyNetrworkCuda(NeuralNetwork& nn, MnistData* trainingSet, int trainingSetSize, MnistData* testSet, int testSetSize, bool notTesting)
{
	NeuralNetworkCuda* d_neuralNetwork;
	NeuralNetworkCuda neuNetCuda;
	float t = 0.71177965666084123452342345f * 0.63584752714172354234581f + 0.13461423452345411116676f * 0.79970759882532455675083f + 0.481727768973546119849898f * 0.08652914346112446344634564634f;
	neuNetCuda.LearningRate = nn.LearningRate;
	neuNetCuda.Type = static_cast<int>(nn.Type);
	neuNetCuda.LearningRateType = static_cast<int>(nn.LearningRateType);
	neuNetCuda.BalanceType = static_cast<int>(nn.BalanceType);
	neuNetCuda.LossFunctionType = static_cast<int>(nn.LossFunctionType);
	neuNetCuda.GradientType = static_cast<int>(nn.GradientType);
	neuNetCuda.Metrics = static_cast<int>(nn.Metrics);
	neuNetCuda.AutoEncoderType = static_cast<int>(nn.AutoEncoderType);
	neuNetCuda.LossCalculation = static_cast<int>(nn.LossCalculation);
	neuNetCuda.LogLoss = static_cast<int>(nn.LogLoss);
	neuNetCuda.BatchSize = nn.BatchSize;
	neuNetCuda.LayersSize = nn.LayersSize;
	neuNetCuda.ro = nn.ro;
	neuNetCuda.iterations = nn.iterations;
	neuNetCuda.beta1Pow = nn.beta1Pow;
	neuNetCuda.beta2Pow = nn.beta2Pow;
	neuNetCuda.betaAELR = nn.betaAELR;

	ERRCHECK(cudaMalloc((void**)&d_neuralNetwork, sizeof(NeuralNetworkCuda)));
	ERRCHECK(cudaMemcpy(d_neuralNetwork, &neuNetCuda, sizeof(NeuralNetworkCuda), cudaMemcpyHostToDevice));
	ERRCHECK(cudaMemcpyToSymbol(neuNet, &d_neuralNetwork, sizeof(NeuralNetworkCuda*)));
	ERRCHECK(cudaDeviceSynchronize());



	LayerCuda* d_layerCuda;
	LayerCuda* layerCuda = new LayerCuda[nn.LayersSize];
	float networkSize = 0;
	for (size_t i = 0; i < nn.LayersSize; i++)
	{
		layerCuda[i].ActivationFunction = static_cast<int>(nn.Layers[i].ActivationFunction);
		layerCuda[i].BatchSize = nn.Layers[i].BatchSize;
		layerCuda[i].DropOutSize = nn.Layers[i].DropOutSize;
		layerCuda[i].IndexVectorSize = nn.Layers[i].IndexVectorSize;
		layerCuda[i].IndexVectorForNextLayerSize = nn.Layers[i].IndexVectorForNextLayerSize;
		layerCuda[i].LayerType = static_cast<int>(nn.Layers[i].LayerType);
		layerCuda[i].Size = nn.Layers[i].Size;
		layerCuda[i].UsingBias = nn.Layers[i].UsingBias;
		layerCuda[i].WeightsSize = nn.Layers[i].WeightsSize;

		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			networkSize += sizeof(float) * nn.Layers[i].Size;
		}
	}
	float* netSize{ 0 };
	std::cout << "net Size = " << networkSize << std::endl;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int maxBatchSize{ 1 };
	int threadsPerBlock{ 0 };
	threadsPerBlock = deviceProp.maxThreadsPerBlock;
	if (notTesting)
	{
		int usedMem = sizeof(float) * trainingSetSize * trainingSet[0].setSize
			+ sizeof(float) * trainingSetSize * trainingSet[0].labelSize
			+ sizeof(float) * testSetSize * testSet[0].setSize
			+ sizeof(float) * testSetSize * testSet[0].labelSize
			+ 1000000000;
		maxBatchSize = (deviceProp.totalGlobalMem - usedMem) / networkSize;
		maxBatchSize = maxBatchSize > testSetSize ? testSetSize : maxBatchSize;
		maxBatchSize = maxBatchSize > nn.BatchSize ? maxBatchSize : nn.BatchSize;
	}
	ERRCHECK(cudaMalloc((void**)&d_layerCuda, sizeof(LayerCuda) * nn.LayersSize));
	ERRCHECK(cudaMemcpy(d_layerCuda, layerCuda, sizeof(LayerCuda) * nn.LayersSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMemcpyToSymbol(layersCd, &d_layerCuda, sizeof(LayerCuda*)));

	int* d_results = NULL, * h_results = NULL;
	h_results = new int[maxBatchSize] {0};
	ERRCHECK(cudaMalloc((void**)&d_results, sizeof(int) * maxBatchSize));
	ERRCHECK(cudaMemcpy(d_results, h_results, sizeof(int) * maxBatchSize, cudaMemcpyHostToDevice));
	copyLayers << <1, 1 >> > (maxBatchSize, d_results);
	ERRCHECK(cudaDeviceSynchronize());


	float* d_weights = NULL, * d_gradients = NULL, * d_gradientslr = NULL, * d_parameters = NULL;
	float* h_weights = NULL, * h_gradients = NULL, * h_gradientslr = NULL, * h_parameters = NULL;
	float* h_tempWeights = NULL, * d_tempWeights = NULL;
	float* d_inputs = NULL, * d_gradientsBatch = NULL, * d_outputs = NULL, * d_targets = NULL;
	float* h_inputs = NULL, * h_gradientsBatch = NULL, * h_outputs = NULL, * h_targets = NULL;
	float* d_inputsHptr = NULL, * d_gradientsBatchHptr = NULL, * d_outputsHptr = NULL, * d_targetsHptr = NULL;
	int* d_idxVec = NULL, * h_idxVec = NULL;
	int* d_nlIdxVec = NULL;
	int* d_neur = NULL, * h_neur = NULL;


	for (size_t i = 0; i < nn.LayersSize; i++)
	{

		int size = nn.Layers[i].Size;
		int weightsSize = nn.Layers[i].WeightsSize;
		int batchSize = maxBatchSize;
		int alocSize = size * sizeof(float);

		ERRCHECK(cudaMalloc((void**)&d_gradients, sizeof(float) * weightsSize));
		h_gradients = new float[weightsSize] {}; for (size_t hi = 0; hi < weightsSize; hi++) h_gradients[hi] = 0;
		ERRCHECK(cudaMemcpy(d_gradients, h_gradients, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_gradientslr, sizeof(float) * weightsSize));
		h_gradientslr = new float[weightsSize] {}; for (size_t hi = 0; hi < weightsSize; hi++) h_gradientslr[hi] = 0;
		ERRCHECK(cudaMemcpy(d_gradientslr, h_gradientslr, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_parameters, sizeof(float) * weightsSize));
		h_parameters = new float[weightsSize] {}; for (size_t hi = 0; hi < weightsSize; hi++) h_parameters[hi] = 0;
		ERRCHECK(cudaMemcpy(d_parameters, h_parameters, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_weights, sizeof(float) * weightsSize));
		ERRCHECK(cudaMemcpy(d_weights, nn.Layers[i].Weights, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_tempWeights, sizeof(float) * weightsSize));
		h_tempWeights = new float[weightsSize] {};
		ERRCHECK(cudaMemcpy(d_tempWeights, h_tempWeights, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_idxVec, sizeof(int) * nn.Layers[i].IndexVectorSize));
		ERRCHECK(cudaMemcpy(d_idxVec, nn.Layers[i].IndexVector, sizeof(int) * nn.Layers[i].IndexVectorSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_nlIdxVec, sizeof(int) * nn.Layers[i].IndexVectorForNextLayerSize));
		ERRCHECK(cudaMemcpy(d_nlIdxVec, nn.Layers[i].IndexVectorForNextLayer, sizeof(int) * nn.Layers[i].IndexVectorForNextLayerSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_neur, sizeof(int) * size));
		h_neur = new int[size];
		for (size_t l = 0; l < size; l++) h_neur[l] = (nn.Layers[i].DropoutNeurons[l] ? 1 : 0);
		ERRCHECK(cudaMemcpy(d_neur, h_neur, sizeof(int) * size, cudaMemcpyHostToDevice));

		auto batchvecSize = batchSize * size;
		auto batchvecAlocSize = batchvecSize * sizeof(float);

		ERRCHECK(cudaMalloc((void**)&d_inputs, batchvecAlocSize));
		h_inputs = new float[batchvecSize] {0.0};
		for (int hi = batchSize; hi--;)
			h_inputs[hi * nn.Layers[i].Size] = nn.Layers[i].UsingBias && nn.Layers[i].LayerType == NeuralEnums::LayerType::HiddenLayer ? 1.0 : 0.0;
		ERRCHECK(cudaMemcpy(d_inputs, h_inputs, batchvecAlocSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_outputs, batchvecAlocSize));
		h_outputs = new float[batchvecSize] {};
		ERRCHECK(cudaMemcpy(d_outputs, i == 0 && !notTesting ? nn.Layers[0].Outputs : h_outputs, batchvecAlocSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_gradientsBatch, batchvecAlocSize));
		h_gradientsBatch = new float[batchvecSize] {};
		ERRCHECK(cudaMemcpy(d_gradientsBatch, h_gradientsBatch, batchvecAlocSize, cudaMemcpyHostToDevice));

		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			ERRCHECK(cudaMalloc((void**)&d_targets, batchvecAlocSize));
			h_targets = new float[batchvecSize] {};
			ERRCHECK(cudaMemcpy(d_targets, i == nn.LayersSize - 1 && !notTesting ? nn.Layers[i].Target : h_targets, batchvecAlocSize, cudaMemcpyHostToDevice));
		}
		initLayer << <1, 1 >> > (
			i,
			d_weights,
			d_tempWeights,
			d_gradients,
			d_gradientslr,
			d_parameters,
			d_idxVec,
			d_nlIdxVec,
			d_neur,
			d_inputs,
			d_outputs,
			d_gradientsBatch,
			d_targets,
			static_cast<int>(nn.Layers[i].LayerType));
		ERRCHECK(cudaDeviceSynchronize());
		auto lastErr = cudaGetLastError();

		delete[] h_tempWeights;
		delete[] h_inputs;
		delete[] h_gradients;
		delete[] h_gradientslr;
		delete[] h_parameters;
		delete[] h_idxVec;
		delete[] h_outputs;
		delete[] h_gradientsBatch;
		delete[] h_neur;
		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			delete[] h_targets;
		}
		ERRCHECK(cudaDeviceSynchronize());
	}
	ERRCHECK(cudaDeviceSynchronize());
	int blocksPerGrid = 0, maxinputs = 0;

	if (notTesting)
	{
		float* h_set = NULL, * h_label = NULL;
		float* h_tset = NULL, * h_tlabel = NULL;
		float* d_set = NULL, * d_label = NULL;
		float* d_testset = NULL, * d_testlabel = NULL;

		h_set = new float[trainingSetSize * trainingSet[0].setSize];
		h_label = new float[trainingSetSize * trainingSet[0].labelSize];

		blocksPerGrid = (trainingSet[0].setSize * trainingSetSize + threadsPerBlock - 1) / threadsPerBlock;

		for (int i = 0; i < trainingSetSize * trainingSet[0].setSize; i++)
			h_set[i] = trainingSet[i / trainingSet[0].setSize].set[i % trainingSet[0].setSize];
		for (int i = 0; i < trainingSetSize * trainingSet[0].labelSize; i++)
			h_label[i] = trainingSet[i / trainingSet[0].labelSize].label[i % trainingSet[0].labelSize];


		ERRCHECK(cudaMalloc((void**)&d_set, sizeof(float) * trainingSet[0].setSize * trainingSetSize));
		ERRCHECK(cudaMemcpy(d_set, h_set, sizeof(float) * trainingSet[0].setSize * trainingSetSize, cudaMemcpyHostToDevice));
		ERRCHECK(cudaMemcpyToSymbol(trainingSetCuda, &d_set, sizeof(float*)));

		ERRCHECK(cudaMalloc((void**)&d_label, sizeof(float) * trainingSet[0].labelSize * trainingSetSize));
		ERRCHECK(cudaMemcpy(d_label, h_label, sizeof(float) * trainingSet[0].labelSize * trainingSetSize, cudaMemcpyHostToDevice));
		ERRCHECK(cudaMemcpyToSymbol(trainingLabelCuda, &d_label, sizeof(float*)));

		h_tset = new float[testSetSize * testSet[0].setSize];
		h_tlabel = new float[testSetSize * testSet[0].labelSize];

		blocksPerGrid = (testSet[0].setSize * testSetSize + threadsPerBlock - 1) / threadsPerBlock;

		for (int i = 0; i < testSetSize * testSet[0].setSize; i++)
			h_tset[i] = testSet[i / testSet[0].setSize].set[i % testSet[0].setSize];
		for (int i = 0; i < testSetSize * testSet[0].labelSize; i++)
			h_tlabel[i] = testSet[i / testSet[0].labelSize].label[i % testSet[0].labelSize];

		ERRCHECK(cudaMalloc((void**)&d_testset, sizeof(float) * testSet[0].setSize * testSetSize));
		ERRCHECK(cudaMemcpy(d_testset, h_tset, sizeof(float) * testSet[0].setSize * testSetSize, cudaMemcpyHostToDevice));
		ERRCHECK(cudaMemcpyToSymbol(testSetCuda, &d_testset, sizeof(float*)));

		ERRCHECK(cudaMalloc((void**)&d_testlabel, sizeof(float) * testSet[0].labelSize * testSetSize));
		ERRCHECK(cudaMemcpy(d_testlabel, h_tlabel, sizeof(float) * testSet[0].labelSize * testSetSize, cudaMemcpyHostToDevice));
		ERRCHECK(cudaMemcpyToSymbol(testLabelCuda, &d_testlabel, sizeof(float*)));

	}
	ERRCHECK(cudaDeviceSynchronize());

	ERRCHECK(cudaDeviceSynchronize());
	auto sdfsdfsdf = cudaGetLastError();


	blocksPerGrid = (nn.BatchSize * nn.Layers[0].Size + threadsPerBlock - 1) / threadsPerBlock;
	maxinputs = nn.Layers[0].Size * nn.BatchSize;

	int globalEpochsCount = 0;
	if (notTesting)
	{
		std::cout << "enter global epochs count" << std::endl;
		std::cin >> globalEpochsCount;
	}
	else
	{
		globalEpochsCount = 1;
	}
	for (int globalEpochs = 0; globalEpochs < globalEpochsCount; globalEpochs++)
	{
		clock_t start = clock();
		setBatchSize << <1, 1 >> > (nn.BatchSize);
		for (size_t i = 0; i < trainingSetSize / nn.BatchSize - 1; i++)
		{
			if (notTesting)
				assignData << <blocksPerGrid, threadsPerBlock >> > (i, maxinputs, false);
			for (size_t layerIdx = 1; layerIdx < nn.LayersSize; layerIdx++)
			{
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * nn.BatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				propagateForward << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);
				//propagateForward << <1, 1 >> > (layerIdx, maxinputs);
				ERRCHECK(cudaDeviceSynchronize());
			}
			for (int layerIdx = nn.LayersSize - 1; layerIdx >= 1; layerIdx--)
			{
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * nn.BatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				backProp << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);
				ERRCHECK(cudaDeviceSynchronize());
				if (nn.BatchSize > 1)
				{
					maxinputs = nn.Layers[layerIdx - 1].IndexVectorForNextLayerSize * nn.Layers[layerIdx].IndexVectorSize;
					blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
					backPropCalcGradients << <blocksPerGrid, threadsPerBlock >> > (layerIdx, nn.BatchSize, maxinputs);
					ERRCHECK(cudaDeviceSynchronize());
				}
			}
			if (i % 1000 == 0)
				std::cout << i << "\r";
			checkLayers << <1, 1 >> > ();
			ERRCHECK(cudaDeviceSynchronize());
		}
		clock_t end = clock();
		std::cout << "Epoch: " << globalEpochs << " training completed in " << (float(end - start) / CLOCKS_PER_SEC) << " seconds. " << std::endl;
		start = clock();
		setBatchSize << <1, 1 >> > (maxBatchSize);
		blocksPerGrid = (maxBatchSize * nn.Layers[0].Size + threadsPerBlock - 1) / threadsPerBlock;
		maxinputs = nn.Layers[0].Size * maxBatchSize;
		for (int i = 0; i < (testSetSize <= maxBatchSize ? 1 : maxBatchSize / testSetSize); i++) //  maxBatchSize % testSetSize !=0 case
		{
			assignData << <blocksPerGrid, threadsPerBlock >> > (i, maxinputs, true);
			ERRCHECK(cudaDeviceSynchronize());
			// Forward propagation
			for (int layerIdx = 1; layerIdx < nn.LayersSize; layerIdx++)
			{
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * maxBatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				propagateForward << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);
				ERRCHECK(cudaDeviceSynchronize());
			}
			checkLayers << <1, 1 >> > ();
			ERRCHECK(cudaDeviceSynchronize());
			maxinputs = maxBatchSize;
			blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
			checkResults << <blocksPerGrid, threadsPerBlock >> > (maxinputs);
			ERRCHECK(cudaDeviceSynchronize());
		}
		checkLayers << <1, 1 >> > ();
		ERRCHECK(cudaDeviceSynchronize());
		countAccuracyPercent << <1, 1 >> > ();
		ERRCHECK(cudaDeviceSynchronize());
		end = clock();
		std::cout << "Epoch: " << globalEpochs << " testing completed in " << (float(end - start) / CLOCKS_PER_SEC) << " seconds. training result: " << passAccuracy << std::endl;
		clearResult << <1, 1 >> > (maxBatchSize);
		ERRCHECK(cudaDeviceSynchronize());
	}
	ERRCHECK(cudaDeviceSynchronize());
	auto ger2r = cudaGetLastError();

	checkLayers << <1, 1 >> > ();
	ERRCHECK(cudaDeviceSynchronize());
	auto err = cudaGetLastError();
}



//checkLayers << <1, 1 >> > ();
//ERRCHECK(cudaDeviceSynchronize());

/*std::stringstream gpu;
gpu << std::fixed << std::setprecision(26) <<passAccuracy;
std::string gpus = gpu.str();*/

//std::stringstream cpu;
//cpu << std::fixed << std::setprecision(26) << t;
//std::string cpus = cpu.str();
// Back propagation
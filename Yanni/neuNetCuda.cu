#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include "enums.h"
#include "neuralNetworkCuda.cuh"
#include "neuralNetwork.h"
#include "layer.h"
#include "mnistData.h"
#include <iostream>
#define ERRCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true, bool wait = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (wait) getch();
#ifdef RELEASE 
		if (abort) exit(code);
#endif
	}
}
__device__ int* results;

__device__ __constant__ NeuralNetworkCuda* neuNet;
__device__ LayerCuda* layersCd;
__device__ MnistData* trainingSetCuda;
__device__ int trainingSetSizeCuda;
__device__ MnistData* testSetCuda;
__device__ int testSetSizeCuda;
__managed__ float passAccuracy;
__device__ float negativeResult;

__global__ void copyDeviceDataElements(float* __restrict__ set, float* __restrict__ label, int idx, bool testSet)
{
	if (!testSet)
	{
		trainingSetCuda[idx].set = set;
		trainingSetCuda[idx].label = label;
		trainingSetCuda[idx].minMax[0] = 0;
		trainingSetCuda[idx].minMax[1] = 0;
	}
	else
	{
		testSetCuda[idx].set = set;
		testSetCuda[idx].label = label;
		testSetCuda[idx].minMax[0] = 0;
		testSetCuda[idx].minMax[1] = 0;
	}
}

__global__ void checkResults(int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < maxinputs)
		results[idx] = GetMaxIndexCuda(neuNet->Layers[neuNet->LayersSize - 1].GetOutputsBatch(idx), neuNet->Layers[neuNet->LayersSize - 1].Size)
		== GetMaxIndexCuda(trainingSetCuda[idx].label, trainingSetCuda[idx].labelSize);
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

__global__ void checkLayers()
{
	passAccuracy = sqrtf(2.0f);
	auto t = neuNet->Layers;
}
__global__ void setBatchSize(int batchSize)
{
	neuNet->BatchSize = batchSize;
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
	int* nlIdxVec,
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
	neuNet->Layers[layerIdx].IndexVectorForNextLayer = nlIdxVec;
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
	int curRow = iter * neuNet->BatchSize + batch;

	if (idx < maxinputs)
	{
		if (!testing)
		{
			neuNet->Layers[0].GetOutputsBatch(batch, j) = trainingSetCuda[curRow].set[j];

			if (j < neuNet->Layers[neuNet->LayersSize - 1].Size)
			{
				neuNet->Layers[neuNet->LayersSize - 1].GetTargetsBatch(batch, j) = trainingSetCuda[curRow].label[j];
			}
		}
		else
		{
			neuNet->Layers[0].GetOutputsBatch(batch, j) = testSetCuda[curRow].set[j];

			if (j < neuNet->Layers[neuNet->LayersSize - 1].Size)
			{
				neuNet->Layers[neuNet->LayersSize - 1].GetTargetsBatch(batch, j) = testSetCuda[curRow].label[j];
			}
		}
	}
}

__global__ void propagateForward(int layerIdx, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[layerIdx].IndexVectorSize;
	int j = idx % neuNet->Layers[layerIdx].IndexVectorSize;
	if (idx < maxinputs)
	{
		neuNet->Layers[layerIdx].CalculateInputs(neuNet->Layers[layerIdx - 1].Size, neuNet->Layers[layerIdx - 1].Outputs, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayer, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayerSize, batch, j, j + 1);
//		neuNet->Layers[layerIdx].CalculateInputs(neuNet->Layers[layerIdx - 1].Size, neuNet->Layers[layerIdx - 1].Outputs, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayer, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayerSize, batch, 0, 1);
		neuNet->Layers[layerIdx].CalculateOutputs(batch, j, j + 1, true, false);
	}
}

__global__ void propagateForwardTest(int layerIdx, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[layerIdx].IndexVectorSize;
	int j = idx % neuNet->Layers[layerIdx].IndexVectorSize;
	if (idx < maxinputs)
	{
		neuNet->Layers[layerIdx].CalculateInputs(neuNet->Layers[layerIdx - 1].Size, neuNet->Layers[layerIdx - 1].Outputs, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayer, neuNet->Layers[layerIdx - 1].IndexVectorForNextLayerSize, batch, j, j + 1);
		neuNet->Layers[layerIdx].CalculateOutputs(batch, j, j + 1, true, false);
	}
}

__global__ void setCurrentLayerSizes(int layerIdx)
{
	neuNet->pLS_ = neuNet->Layers[layerIdx - 1].Size;
	neuNet->biasShift_ = neuNet->Layers[layerIdx].UsingBias ? 1 : 0;
	neuNet->curLayerSize = neuNet->Layers[layerIdx].Size;
}

__global__ void backProp(int layerIdx, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int batch = idx / neuNet->Layers[layerIdx].IndexVectorSize;
	int j = idx % neuNet->Layers[layerIdx].IndexVectorSize;
	if (idx < maxinputs)
	{
		neuNet->PropagateBack(layerIdx, batch, j, j + 1);
	}
}

__global__ void backPropCalcGradients(int layerIdx, int batchCount, int maxinputs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < maxinputs)
	{
		neuNet->PropagateBackCalculateGradients(layerIdx, batchCount, idx, idx + 1);
		//neuNet->PropagateBackCalculateGradients(layerIdx, batchCount, 0, 1000);
	}
}


/// <summary>
/// 
/// </summary>
/// <param name="nn"> - Neural network</param>
/// <param name="trainingSet"> - Training set</param>
/// <param name="trainingSetSize"> - Training set size</param>
/// <param name="testSet"> - Test set</param>
/// <param name="testSetSize"> - Test set size</param>
/// <param name="copyData"> - For testing puproses - false when testing</param>
void copyNetrworkCuda(NeuralNetwork& nn, MnistData* trainingSet, int trainingSetSize, MnistData* testSet, int testSetSize, bool notTesting)
{
	ERRCHECK(cudaDeviceSynchronize());

	NeuralNetworkCuda* d_neuralNetwork;
	NeuralNetworkCuda neuNetCuda;

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
	for (int i = 0; i < nn.LayersSize; i++)
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
		networkSize += sizeof(float) * nn.Layers[i].Size * 3;
		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			networkSize += sizeof(float) * nn.Layers[i].Size;
		}
	}
	float* netSize = 0;
	std::cout << "net Size = " << networkSize << std::endl;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int usedMem = sizeof(float) * trainingSetSize * trainingSet[0].setSize
		+ sizeof(float) * trainingSetSize * trainingSet[0].labelSize
		+ sizeof(float) * testSetSize * testSet[0].setSize
		+ sizeof(float) * testSetSize * testSet[0].labelSize
		+ 1000000000;
	int maxBatchSize = (deviceProp.totalGlobalMem - usedMem) / networkSize;
	maxBatchSize = maxBatchSize > testSetSize ? testSetSize : maxBatchSize;
	maxBatchSize = maxBatchSize > nn.BatchSize ? maxBatchSize : nn.BatchSize;
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
	float* d_inputsBatch = NULL, * d_gradientsBatch = NULL, * d_outputsBatch = NULL, * d_targetsBatch = NULL;
	float* h_inputsBatch = NULL, * h_gradientsBatch = NULL, * h_outputsBatch = NULL, * h_targetsBatch = NULL;
	float* d_inputsBatchHptr = NULL, * d_gradientsBatchHptr = NULL, * d_outputsBatchHptr = NULL, * d_targetsBatchHptr = NULL;
	int* d_idxVec = NULL;
	int* d_nlIdxVec = NULL;
	int* d_neur = NULL, * h_neur = NULL;


	for (int i = 0; i < nn.LayersSize; i++)
	{

		int size = nn.Layers[i].Size;
		int weightsSize = nn.Layers[i].WeightsSize;
		int batchSize = maxBatchSize;
		int alocSize = size * sizeof(float);

		ERRCHECK(cudaMalloc((void**)&d_gradients, sizeof(float) * weightsSize));
		h_gradients = new float[weightsSize] {};
		ERRCHECK(cudaMemcpy(d_gradients, h_gradients, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_gradientslr, sizeof(float) * weightsSize));
		h_gradientslr = new float[weightsSize] {};
		ERRCHECK(cudaMemcpy(d_gradientslr, h_gradientslr, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_parameters, sizeof(float) * weightsSize));
		h_parameters = new float[weightsSize] {};
		ERRCHECK(cudaMemcpy(d_parameters, h_parameters, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_tempWeights, sizeof(float) * weightsSize));
		h_tempWeights = new float[weightsSize] {};
		ERRCHECK(cudaMemcpy(d_tempWeights, h_tempWeights, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_weights, sizeof(float) * weightsSize));
		ERRCHECK(cudaMemcpy(d_weights, nn.Layers[i].Weights, sizeof(float) * weightsSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_idxVec, sizeof(int) * nn.Layers[i].IndexVectorSize));
		ERRCHECK(cudaMemcpy(d_idxVec, nn.Layers[i].IndexVector, sizeof(int) * nn.Layers[i].IndexVectorSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_nlIdxVec, sizeof(int) * nn.Layers[i].IndexVectorForNextLayerSize));
		ERRCHECK(cudaMemcpy(d_nlIdxVec, nn.Layers[i].IndexVectorForNextLayer, sizeof(int) * nn.Layers[i].IndexVectorForNextLayerSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_neur, sizeof(int) * size));
		h_neur = new int[size] {0};
		ERRCHECK(cudaMemcpy(d_neur, h_neur, sizeof(int) * size, cudaMemcpyHostToDevice));

		auto batchvecSize = batchSize * size;
		auto batchvecAlocSize = batchvecSize * sizeof(float);

		ERRCHECK(cudaMalloc((void**)&d_inputsBatch, batchvecAlocSize));
		h_inputsBatch = new float[batchvecSize] {0.0};
		for (int hi = batchSize; hi--;)
			h_inputsBatch[hi * nn.Layers[i].Size] = nn.Layers[i].UsingBias && nn.Layers[i].LayerType == NeuralEnums::LayerType::HiddenLayer ? 1.0 : 0.0;
		ERRCHECK(cudaMemcpy(d_inputsBatch, h_inputsBatch, batchvecAlocSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_outputsBatch, batchvecAlocSize));
		h_outputsBatch = new float[batchvecSize] {};
		ERRCHECK(cudaMemcpy(d_outputsBatch, h_outputsBatch, batchvecAlocSize, cudaMemcpyHostToDevice));

		ERRCHECK(cudaMalloc((void**)&d_gradientsBatch, batchvecAlocSize));
		h_gradientsBatch = new float[batchvecSize] {};
		ERRCHECK(cudaMemcpy(d_gradientsBatch, h_gradientsBatch, batchvecAlocSize, cudaMemcpyHostToDevice));

		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			ERRCHECK(cudaMalloc((void**)&d_targetsBatch, batchvecAlocSize));
			h_targetsBatch = new float[batchvecSize] {};
			ERRCHECK(cudaMemcpy(d_targetsBatch, h_targetsBatch, batchvecAlocSize, cudaMemcpyHostToDevice));
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
			d_inputsBatch,
			d_outputsBatch,
			d_gradientsBatch,
			d_targetsBatch,
			static_cast<int>(nn.Layers[i].LayerType));
		ERRCHECK(cudaDeviceSynchronize());
		auto lastErr = cudaGetLastError();

		delete[] h_inputsBatch;
		delete[] h_tempWeights;
		delete[] h_gradients;
		delete[] h_gradientslr;
		delete[] h_parameters;
		delete[] h_outputsBatch;
		delete[] h_gradientsBatch;
		delete[] h_neur;
		if (nn.Layers[i].LayerType == NeuralEnums::LayerType::OutputLayer)
		{
			delete[] h_targetsBatch;
		}
		ERRCHECK(cudaDeviceSynchronize());
	}

	ERRCHECK(cudaDeviceSynchronize());
	MnistData* d_mnist, * d_mnistTest;

	auto ertr = cudaGetLastError();

	ERRCHECK(cudaMalloc((void**)&d_mnist, sizeof(MnistData) * trainingSetSize));
	ERRCHECK(cudaMemcpy(d_mnist, trainingSet, sizeof(MnistData) * trainingSetSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMemcpyToSymbol(trainingSetCuda, &d_mnist, sizeof(MnistData*)));
	ERRCHECK(cudaDeviceSynchronize());

	ERRCHECK(cudaMalloc((void**)&d_mnistTest, sizeof(MnistData) * testSetSize));
	ERRCHECK(cudaMemcpy(d_mnistTest, testSet, sizeof(MnistData) * testSetSize, cudaMemcpyHostToDevice));
	ERRCHECK(cudaMemcpyToSymbol(testSetCuda, &d_mnistTest, sizeof(MnistData*)));
	ERRCHECK(cudaDeviceSynchronize());

	float* d_set = NULL, * d_label = NULL;
	float* d_testset = NULL, * d_testlabel = NULL;
	if (notTesting)
	{
		for (int i = 0; i < trainingSetSize; i++)
		{
			ERRCHECK(cudaMalloc((void**)&d_set, sizeof(float) * trainingSet[i].setSize));
			ERRCHECK(cudaMalloc((void**)&d_label, sizeof(float) * trainingSet[i].labelSize));
			ERRCHECK(cudaMemcpy(d_set, trainingSet[i].set, sizeof(float) * trainingSet[i].setSize, cudaMemcpyHostToDevice));
			ERRCHECK(cudaMemcpy(d_label, trainingSet[i].label, sizeof(float) * trainingSet[i].labelSize, cudaMemcpyHostToDevice));
			copyDeviceDataElements << <1, 1 >> > (d_set, d_label, i, false);
		}
		for (int i = 0; i < testSetSize; i++)
		{
			ERRCHECK(cudaMalloc((void**)&d_testset, sizeof(float) * testSet[i].setSize));
			ERRCHECK(cudaMalloc((void**)&d_testlabel, sizeof(float) * testSet[i].labelSize));
			ERRCHECK(cudaMemcpy(d_testset, testSet[i].set, sizeof(float) * testSet[i].setSize, cudaMemcpyHostToDevice));
			ERRCHECK(cudaMemcpy(d_testlabel, testSet[i].label, sizeof(float) * testSet[i].labelSize, cudaMemcpyHostToDevice));
			copyDeviceDataElements << <1, 1 >> > (d_testset, d_testlabel, i, true);
		}
	}
	ERRCHECK(cudaDeviceSynchronize());

	checkLayers << <1, 1 >> > ();
	ERRCHECK(cudaDeviceSynchronize());
	auto sdfsdfsdf = cudaGetLastError();
	int threadsPerBlock = deviceProp.maxThreadsPerBlock;
	int globalEpochsCount = 1;
	std::cout << "enter global epochs count" << std::endl;
	std::cin >> globalEpochsCount;
	for (int globalEpochs = 0; globalEpochs < globalEpochsCount; globalEpochs++)
	{
		clock_t start = clock();
		int blocksPerGrid = (nn.BatchSize * nn.Layers[0].Size + threadsPerBlock - 1) / threadsPerBlock;
		int maxinputs = nn.Layers[0].Size * nn.BatchSize;

		//training cycle
		setBatchSize << <1, 1 >> > (nn.BatchSize);
		for (int i = 0; i < trainingSetSize / nn.BatchSize - 1; i++)
		{
			if (notTesting)
				assignData << <blocksPerGrid, threadsPerBlock >> > (i, maxinputs, false);
			ERRCHECK(cudaDeviceSynchronize());
			// Forward propagation
			for (int layerIdx = 1; layerIdx < nn.LayersSize; layerIdx++)
			{
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * nn.BatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				propagateForward << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);
				//propagateForward << <1, 1>> > (layerIdx, maxinputs);
				ERRCHECK(cudaDeviceSynchronize());
			}
			checkLayers << <1, 1 >> > ();
			ERRCHECK(cudaDeviceSynchronize());
			sdfsdfsdf = cudaGetLastError();
			// Back propagation
			for (int layerIdx = nn.LayersSize - 1; layerIdx >= 1; layerIdx--)
			{
				setCurrentLayerSizes << <1, 1 >> > (layerIdx);
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * nn.BatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				backProp << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);

				if (notTesting)
				{
					maxinputs = nn.Layers[layerIdx - 1].IndexVectorForNextLayerSize * nn.Layers[layerIdx].IndexVectorSize;
					blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
					backPropCalcGradients << <blocksPerGrid, threadsPerBlock >> > (layerIdx, nn.BatchSize, maxinputs);
				}
			}
		}
		checkLayers << <1, 1 >> > ();
		ERRCHECK(cudaDeviceSynchronize());
		ERRCHECK(cudaDeviceSynchronize());
		auto ger2r = cudaGetLastError();
		//test cycle
		setBatchSize << <1, 1 >> > (maxBatchSize);
		blocksPerGrid = (maxBatchSize * nn.Layers[0].Size + threadsPerBlock - 1) / threadsPerBlock;
		maxinputs = nn.Layers[0].Size * maxBatchSize;
		for (int i = 0; i < (testSetSize <= maxBatchSize ? 1 : maxBatchSize / testSetSize); i++) //  maxBatchSize % testSetSize !=0 case
		{
			assignData << <blocksPerGrid, threadsPerBlock >> > (i, maxinputs, true);
			ERRCHECK(cudaDeviceSynchronize());
			checkLayers << <1, 1 >> > ();
			ERRCHECK(cudaDeviceSynchronize());
			// Forward propagation
			for (int layerIdx = 1; layerIdx < nn.LayersSize; layerIdx++)
			{
				maxinputs = nn.Layers[layerIdx].IndexVectorSize * maxBatchSize;
				blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
				propagateForward << <blocksPerGrid, threadsPerBlock >> > (layerIdx, maxinputs);
				ERRCHECK(cudaDeviceSynchronize());
			}
			maxinputs = maxBatchSize;
			blocksPerGrid = (maxinputs + threadsPerBlock - 1) / threadsPerBlock;
			checkResults << <blocksPerGrid, threadsPerBlock >> > (maxinputs);
			ERRCHECK(cudaDeviceSynchronize());
		}
		checkLayers << <1, 1 >> > ();
		ERRCHECK(cudaDeviceSynchronize());
		countAccuracyPercent << <1, 1 >> > ();
		ERRCHECK(cudaDeviceSynchronize());
		clock_t end = clock();
		std::cout << "Epoch: " << globalEpochs << " completed in " << (float(end - start) / CLOCKS_PER_SEC) << " seconds. training result: " << passAccuracy << std::endl;
		clearResult << <1, 1 >> > (maxBatchSize);
		ERRCHECK(cudaDeviceSynchronize());
	}
	ERRCHECK(cudaDeviceSynchronize());
	auto ger2r = cudaGetLastError();

	checkLayers << <1, 1 >> > ();
	ERRCHECK(cudaDeviceSynchronize());
	auto err = cudaGetLastError();
	int t = 10;
}



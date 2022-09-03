#include "Layer.h"
#include "activationFunctions.h"
#include <cstddef>

Layer::Layer(int size, NeuralEnums::LayerType layerType, NeuralEnums::ActivationFunction activationFunction, float bias, float dropoutSize, int batchSize)
{
	Size = size;
	LayerType = layerType;
	ActivationFunction = activationFunction;
	Inputs = new float[size] {};
	Outputs = new float[size] {};
	DropoutNeurons = new bool[size];
	Parameters = new float[size] {};
	RoHat = new float[size] {};
	GradientsForGrads = new float[size] {};
	DropOutSize = dropoutSize;
	UsingBias = !(bias == NULL);
	BatchSize = batchSize;



	auto biasShift = UsingBias ? 1 : 0;
	if (layerType == NeuralEnums::LayerType::InputLayer || layerType == NeuralEnums::LayerType::OutputLayer)
		biasShift = 0;//input and output layers should not drop perceprton
	IndexVectorSize = Size - biasShift;
	IndexVector = new int[IndexVectorSize] {};
	IndexVectorForNextLayer = new int[Size] {};
	IndexVectorForNextLayerSize = Size;
	IndexVectorForNextLayer[0] = 0;
	for (int i = biasShift; i < Size; i++)
	{
		IndexVector[i - biasShift] = i;
		IndexVectorForNextLayer[i] = i;
	}

	if (batchSize > 1)
	{
		InputsBatch = new float* [batchSize];
		InputsBatch = new float* [batchSize];
		OutputsBatch = new float* [batchSize];
		if (LayerType == NeuralEnums::LayerType::OutputLayer)
			TargetsBatch = new float* [batchSize];

		for (int i = 0; i < batchSize; i++)
		{
			InputsBatch[i] = new float[size] {};
			OutputsBatch[i] = new float[size] {};
			if (LayerType == NeuralEnums::LayerType::OutputLayer)
			{
				TargetsBatch = new float* [batchSize];
				TargetsBatch[i] = new float[size] {};
			}
		}
	}
	for (size_t i = 0; i < Size; i++)
	{
		DropoutNeurons[i] = false;
	}
	if (DropOutSize > 0)
	{
		for (int i = 0; i < Size; i++)
			DropoutNeurons[i] = i < Size* DropOutSize;
	}
	if (UsingBias)
	{
		Inputs[0] = bias;
		if (batchSize > 1)
			for (int i = 0; i < batchSize; i++)
			{
				InputsBatch[i] = new float[Size] {};
				InputsBatch[i][0] = bias;
			}
	}
}

//---------------------------------------------------
void Layer::CalculateInputsThreaded(
	float* prevLayerOutput,
	int prevLayerSize,
	float** prevLayerOutputBatch,
	int* prevLayerIndex,
	int& prevLayerIndexVectorSize,
	bool& training,
	int& numThreads,
	std::vector<WorkerThread*>& _workers)
{

	if (BatchSize > 1 && training)
	{
		int chunkSize = BatchSize / numThreads;
		int idx = 0;
		for (int i = 0; i < numThreads; i++)
		{
			idx = chunkSize * i;
			_workers[i]->doAsync(std::bind(&Layer::CalcInputsDelegate, this,
				prevLayerOutput, prevLayerSize, prevLayerOutputBatch, prevLayerIndex, prevLayerIndexVectorSize, training, idx, idx + chunkSize));
		}
		for (int k = 0; k < numThreads; k++)
			_workers[k]->wait();
	}
	else
	{
		int chunkSize = IndexVectorSize / numThreads == 0 ? 1 : IndexVectorSize / numThreads;
		int iterator = numThreads > IndexVectorSize ? IndexVectorSize : numThreads;

		for (int i = iterator; i--;)
		{
			int start = i * chunkSize;
			int end = (i + 1) == iterator ? IndexVectorSize : (i + 1) * chunkSize;

			_workers[i]->doAsync(std::bind(&Layer::CalcInputsDelegate, this, prevLayerOutput, prevLayerSize, prevLayerOutputBatch, prevLayerIndex, prevLayerIndexVectorSize, training, start, end));
		}
		for (int i = iterator; i--;)
			_workers[i]->wait();
	}
}

//---------------------------------------------------
void Layer::CalculateOutputsThreaded(int& numThreads, bool& training, bool& countingRohat, std::vector<WorkerThread*>& _workers)
{
	if (BatchSize > 1 && training)
	{
		bool miniBatch = BatchSize > 1 && training;
		int chunkSize = BatchSize / numThreads;
		int idx = 0;

		for (int i = numThreads; i--;)
		{
			idx = chunkSize * i;
			_workers[i]->doAsync(std::bind(&Layer::CalcOutputsDelegate, this, idx, idx + chunkSize, training, countingRohat));
		}
		for (int k = numThreads; k--;)
			_workers[k]->wait();
	}
	else
	{
		int chunkSize = IndexVectorSize / numThreads == 0 ? 1 : IndexVectorSize / numThreads;
		int iterator = numThreads > IndexVectorSize ? IndexVectorSize : numThreads;

		for (int i = iterator; i--;)
		{
			int start = i * chunkSize;
			int end = (i + 1) == iterator ? IndexVectorSize : (i + 1) * chunkSize;
			_workers[i]->doAsync(std::bind(&Layer::CalcOutputsDelegate, this, start, end, training, countingRohat));
		}
		for (int i = iterator; i--;)
			_workers[i]->wait();
	}
}

void Layer::CalcInputsDelegate(
	float* prevLayerOutput,
	int prevLayerSize,
	float** prevLayerOutputBatch,
	int* prevLayerIndexes,
	int& prevLayerIndexVectorSize,
	bool& training,
	int& start, int& end)
{
	float result;
	bool drop;
	int biasShift = UsingBias ? 1 : 0;
	int k = 0, i = 0;
	if (BatchSize > 1 && training)
	{
		for (int batch = start; batch < end; batch++)
		{

			for (int kk = IndexVectorSize; kk--;)
			{
				k = IndexVector[kk];
				result = 0.0;
				for (int ii = prevLayerIndexVectorSize; ii--;)
				{
					i = prevLayerIndexes[ii];
					result += prevLayerOutputBatch[batch][i] * Weights[(k - biasShift) * prevLayerSize + i];
				}
				InputsBatch[batch][k] = result;
			}
		}
	}
	else
	{
		for (int kk = start; kk < end; kk++)
		{
			k = IndexVector[kk];
			result = 0.0;
			for (int ii = prevLayerIndexVectorSize; ii--;)
			{
				i = prevLayerIndexes[ii];
				result += prevLayerOutput[i] * Weights[(k - biasShift) * prevLayerSize + i];
			}
			Inputs[k] = result;
		}
	}
}

//---------------------------------------------------
void Layer::CalcOutputsDelegate(int& start, int& end, bool& training, bool& countingRohat)
{
	//TODO ჩასამატებელია SoftMax რეალიზაცია 
	if (BatchSize > 1 && training)
	{
		for (int batch = start; batch < end; batch++)
		{
			int vStart = 0, vEnd = IndexVectorSize;
			ActivateWith(InputsBatch[batch], OutputsBatch[batch], IndexVector, vStart, vEnd, ActivationFunction);
			if (UsingBias)
				OutputsBatch[batch][0] = InputsBatch[batch][0];
		}
	}
	else
	{
		ActivateWith(Inputs, Outputs, IndexVector, start, end, ActivationFunction);
		if (countingRohat)
			for (int i = start; i < end; i++)
				RoHat[i] += Outputs[i];
		if (start == 0 && UsingBias)
			Outputs[0] = Inputs[0];
	}
}

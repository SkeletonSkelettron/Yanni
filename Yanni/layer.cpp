#include "layer.h"
#include "activationFunctions.h"
#include <cstddef>

Layer::Layer(size_t size, NeuralEnums::LayerType layerType, NeuralEnums::ActivationFunction activationFunction, float bias, float dropoutSize, int batchSize)
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
	IndexVector = new size_t[IndexVectorSize] {};
	IndexVectorForNextLayer = new size_t[Size] {};
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
	size_t prevLayerSize,
	float** prevLayerOutputBatch,
	size_t* prevLayerIndex,
	size_t& prevLayerIndexVectorSize,
	bool& training,
	size_t& numThreads,
	std::vector<WorkerThread*>& _workers)
{

	if (BatchSize > 1 && training)
	{
		size_t chunkSize = BatchSize / numThreads;
		size_t idx = 0;
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
		size_t chunkSize = IndexVectorSize / numThreads == 0 ? 1 : IndexVectorSize / numThreads;
		size_t iterator = numThreads > IndexVectorSize ? IndexVectorSize : numThreads;

		for (size_t i = iterator; i--;)
		{
			size_t start = i * chunkSize;
			size_t end = (i + 1) == iterator ? IndexVectorSize : (i + 1) * chunkSize;

			_workers[i]->doAsync(std::bind(&Layer::CalcInputsDelegate, this, prevLayerOutput, prevLayerSize, prevLayerOutputBatch, prevLayerIndex, prevLayerIndexVectorSize, training, start, end));
		}
		for (size_t i = iterator; i--;)
			_workers[i]->wait();
	}
}

//---------------------------------------------------
void Layer::CalculateOutputsThreaded(size_t& numThreads, bool& training, bool& countingRohat, std::vector<WorkerThread*>& _workers)
{
	if (BatchSize > 1 && training)
	{
		size_t miniBatch = BatchSize > 1 && training;
		size_t chunkSize = BatchSize / numThreads;
		size_t idx = 0;

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
		size_t chunkSize = IndexVectorSize / numThreads == 0 ? 1 : IndexVectorSize / numThreads;
		size_t iterator = numThreads > IndexVectorSize ? IndexVectorSize : numThreads;

		for (size_t i = iterator; i--;)
		{
			size_t start = i * chunkSize;
			size_t end = (i + 1) == iterator ? IndexVectorSize : (i + 1) * chunkSize;
			_workers[i]->doAsync(std::bind(&Layer::CalcOutputsDelegate, this, start, end, training, countingRohat));
		}
		for (size_t i = iterator; i--;)
			_workers[i]->wait();
	}
}

void Layer::CalcInputsDelegate(
	float* prevLayerOutput, 
	size_t prevLayerSize, 
	float** prevLayerOutputBatch, 
	size_t* prevLayerIndexes, 
	size_t& prevLayerIndexVectorSize, 
	bool& training, 
	size_t& start, size_t& end)
{
	float result;
	size_t biasShift = UsingBias ? 1 : 0;
	size_t k = 0, i = 0;
	if (BatchSize > 1 && training)
	{
		for (size_t batch = start; batch < end; batch++)
		{

			for (size_t kk = IndexVectorSize; kk--;)
			{
				k = IndexVector[kk];
				result = 0.0;
				for (size_t ii = prevLayerIndexVectorSize; ii--;)
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
		for (size_t kk = start; kk < end; kk++)
		{
			k = IndexVector[kk];
			result = 0.0;
			for (size_t ii = prevLayerIndexVectorSize; ii--;)
			{
				i = prevLayerIndexes[ii];
				result += prevLayerOutput[i] * Weights[(k - biasShift) * prevLayerSize + i];
			}
			Inputs[k] = result;
		}
	}
}

//---------------------------------------------------
void Layer::CalcOutputsDelegate(size_t& start, size_t& end, bool& training, bool& countingRohat)
{
	//TODO ჩასამატებელია SoftMax რეალიზაცია 
	if (BatchSize > 1 && training)
	{
		for (size_t batch = start; batch < end; batch++)
		{
			size_t vStart = 0, vEnd = IndexVectorSize;
			ActivateWith(InputsBatch[batch], OutputsBatch[batch], IndexVector, vStart, vEnd, ActivationFunction);
			if (UsingBias)
				OutputsBatch[batch][0] = InputsBatch[batch][0];
		}
	}
	else
	{
		ActivateWith(Inputs, Outputs, IndexVector, start, end, ActivationFunction);
		if (countingRohat)
			for (size_t i = start; i < end; i++)
				RoHat[i] += Outputs[i];
		if (start == 0 && UsingBias)
			Outputs[0] = Inputs[0];
	}
}

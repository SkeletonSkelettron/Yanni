#ifndef LAYER_H
#define LAYER_H
#include "Enums.h"
#include "WorkerThread.h"
#include <vector>
class Layer
{
public:
	size_t Size;
	float* Inputs;
	float** InputsBatch;
	float* Weights;
	float* TempWeights;
	float* MultipliedSums;
	int WeightsSize;
	float* Outputs;
	float* RoHat;
	float** RoHatBatch;
	float** OutputsBatch;
	float* Gradients;
	float* GradientsLR;
	float** GradientsBatch;
	float* Parameters;
	float* GradientsForGrads;
	float* LearningRates;
	float* Target;
	float** TargetsBatch;
	size_t* IndexVector;
	size_t IndexVectorSize;
	size_t* IndexVectorForNextLayer;
	size_t IndexVectorForNextLayerSize;
	float DropOutSize;
	bool UsingBias;
	bool* DropoutNeurons; // if DropoutNeurons[i]==true, then Inputs[i] and Outputs[0]  do not take part in calculations
	int BatchSize;
	NeuralEnums::ActivationFunction ActivationFunction;
	NeuralEnums::LayerType LayerType;
	Layer() {}
	Layer(size_t size, NeuralEnums::LayerType layerType, NeuralEnums::ActivationFunction activationFunction, float bias, float dropoutSize = 0, int batchSize = 1);

	void CalcInputsDelegate(
		float* prevLayerOutput,
		size_t prevLayerSize,
		float** prevLayerOutputBatch,
		size_t* prevLayerIndexes,
		size_t& prevLayerIndexVectorSize,
		bool& training, size_t& start, size_t& end);
	void CalcOutputsDelegate(size_t& start, size_t& end, bool& training, bool& countingRohat);
	void CalculateInputsThreaded(
		float* prevLayerOutput,
		size_t prevLayerSize,
		float** prevLayerOutputBatch,
		size_t* prevLayerIndex,
		size_t& prevLayerIndexVectorSize,
		bool& training,
		size_t& numThreads,
		std::vector<WorkerThread*>& _workers);
	void CalculateOutputsThreaded(size_t& numThreads, bool& training, bool& countingRohat, std::vector<WorkerThread*>& _workers);
};
#endif
//Layer.h

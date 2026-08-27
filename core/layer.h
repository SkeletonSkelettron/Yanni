#ifndef LAYER_H
#define LAYER_H
#include "enums.h"
#include "workerThread.h"
#include <vector>
class Layer {
public:
  size_t Size;
  float *Inputs;
  float **InputsBatch;
  float *Weights;
  float *TempWeights;
  float *MultipliedSums;
  int WeightsSize;
  float *Outputs;
  float *RoHat;
  float **RoHatBatch;
  float **OutputsBatch;
  float *Gradients;
  float *GradientsLR;
  float **GradientsBatch;
  float *Parameters;
  float *GradientsForGrads;
  float *LearningRates;
  float *Target;
  float **TargetsBatch;
  int *IndexVector;
  int IndexVectorSize;
  int *IndexVectorForNextLayer;
  int IndexVectorForNextLayerSize;
  float DropOutSize;
  bool UsingBias;
  float *DropoutNeurons;
  int BatchSize;

  NeuralEnums::ActivationFunction ActivationFunction;
  NeuralEnums::LayerType LayerType;
  Layer() {}
  Layer(int size, NeuralEnums::LayerType layerType,
        NeuralEnums::ActivationFunction activationFunction, float bias,
        float dropoutSize = 0, int batchSize = 1);

  void CalcInputsDelegate(float *prevLayerOutput, int prevLayerSize,
                          float **prevLayerOutputBatch, bool &training,
                          int &start, int &end);
  void CalcOutputsDelegate(int &start, int &end, bool &training,
                           bool &countingRohat);
  void CalculateInputsThreaded(float *prevLayerOutput, int prevLayerSize,
                               float **prevLayerOutputBatch, int &numThreads,
                               std::vector<WorkerThread *> &_workers);
  void CalculateOutputsThreaded(int &numThreads, bool &training,
                                bool &countingRohat,
                                std::vector<WorkerThread *> &_workers);
};
#endif
// Layer.h

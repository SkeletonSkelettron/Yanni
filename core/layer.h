#ifndef LAYER_H
#define LAYER_H
#include "enums.h"
#include "workerThread.h"
#include <algorithm>
#include <vector>

class Layer {
public:
  int Size;
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
  bool UsingBias;
  float *Mask;
  int BatchSize;
  float DropOutSize;

  NeuralEnums::ActivationFunction ActivationFunction;
  NeuralEnums::LayerType LayerType;
  Layer() {}
  Layer(int size, NeuralEnums::LayerType layerType,
        NeuralEnums::ActivationFunction activationFunction, float bias,
        float dropOutSize = 0.0f, int batchSize = 1);

  void CalcInputsDelegate(float *prevLayerOutput, int prevLayerSize,
                          float **prevLayerOutputBatch, bool &training,
                          int &start, int &end);
  void CalcOutputsDelegate(int &start, int &end, bool &training,
                           bool &countingRohat);
  void CalculateInputsThreaded(float *prevLayerOutput, int prevLayerSize,
                               float **prevLayerOutputBatch, bool &training,
                               int &numThreads,
                               std::vector<WorkerThread *> &_workers);
  void CalculateOutputsThreaded(int &numThreads, bool &training,
                                bool &countingRohat,
                                std::vector<WorkerThread *> &_workers);
};
#endif
// Layer.h

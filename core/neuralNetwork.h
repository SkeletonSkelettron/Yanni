#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "../functions/gradientFunctions.h"
#include "../functions/statisticFunctions.h"
#include "../functions/activationFunctions.h"
#include "enums.h"
#include "layer.h"
#include "../functions/lossFunctions.h"
#include "workerThread.h"
#include <algorithm>
#include <cstdio>
#include <math.h>
#include <vector>
class NeuralNetwork {
public:
  int ThreadCount;
  float LearningRate;
  std::vector<WorkerThread *> workers;
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
  Layer *Layers;
  int LayersSize;
  float ***GradientsTemp;
  float *lossesTmp;
  float ro;
  int iterations = 0;
  float beta1Pow = 0.9;
  float beta2Pow = 0.999;
  float betaAELR = 0.001;

  const float momentum = 0.9;
  const float epsilon = 0.0000001;
  const float startingLearningRate = 0.001;
  const float beta1 = 0.9;
  const float beta2 = 0.999;

  // weight decay parameter for sparce autoencoder
  float lambda = 0.7;
  void NeuralNetworkInit();
  void InitializeWeights();
  void PrepareForTesting();
  // ერთხელ ითვლება NeuralNetworkInit-ში: აქვს თუ არა რომელიმე ლეიერს dropout.
  // თუ არა, ShuffleDropoutsPlain ყოველ პაკეტზე უქმად აღარ გამოიძახება.
  bool UsingDropout = false;
  float PropagateForwardThreaded(bool training, bool countingRohat);
  void PropagateBackThreaded();
  void PropagateBackDelegate(int i, int start, int end);
  void PropagateBackDelegateNew(int i, int start, int end);
  void PropagateBackDelegateBatch(int start, int end, int threadNum);
  void ShuffleDropoutsPlain();
  void CalculateWeightsBatch();
  void CalculateWeightsBatchSub(int i, int start, int end);
  float CalculateLoss(bool &training);
  void CalculateLossSub(int start, int end, int klbstart, int klbend,
                        float &loss);
  void CalculateLossBatchSub(int start, int end, float &loss);

  float GetLearningRateMultipliedByGrad(float &gradient, int &iterator, int &j);
  float Adam(float &gradient, int &j, int &iterator);
  float AdaGrad(float *gradients, float &gradient, int &j);
  float AdaDelta(float *gradients, float *parameters, float &Gradient, int &j);
  float AdamMod(float &Gradient, int &j, int &iterator);
  float AdaMax(float &gradient, int &j, int &iterator);
  float RMSProp(float *gradients, float &gradient, int &j);
};

// Dump the assembled network (not the JSON) before training starts.
void PrintNetworkInfo(NeuralNetwork &nn, size_t trainingSetSize);
#endif

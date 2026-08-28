#include "layer.h"
#include "../functions/activationFunctions.h"

Layer::Layer(int size, NeuralEnums::LayerType layerType,
             NeuralEnums::ActivationFunction activationFunction, float bias,
             int batchSize) {

  UsingBias = !(bias == 0.0f);
  Size = size + (UsingBias ? 1 : 0);
  LayerType = layerType;
  ActivationFunction = activationFunction;
  Mask = new float[Size]{};

  Inputs = new float[Size]{};
  Outputs = new float[Size]{};
  std::fill(Mask, Mask + Size, 1.0f);
  RoHat = new float[Size]{};
  GradientsForGrads = new float[Size]{};
  BatchSize = batchSize;

  if (batchSize > 1) {
    InputsBatch = new float *[batchSize];
    OutputsBatch = new float *[batchSize];
    if (LayerType == NeuralEnums::LayerType::OutputLayer)
      TargetsBatch = new float *[batchSize];

    for (int i = 0; i < batchSize; i++) {
      InputsBatch[i] = new float[Size]{};
      OutputsBatch[i] = new float[Size]{};
      if (LayerType == NeuralEnums::LayerType::OutputLayer)
        TargetsBatch[i] = new float[Size]{};
    }
    if (UsingBias) {
      for (int i = 0; i < batchSize; i++) {
        InputsBatch[i][0] = bias;
      }
    }
  }
  if (UsingBias) {
    Inputs[0] = bias;
  }
}

//---------------------------------------------------
void Layer::CalculateInputsThreaded(float *prevLayerOutput, int prevLayerSize,
                                    float **prevLayerOutputBatch,
                                    bool &training, int &numThreads,
                                    std::vector<WorkerThread *> &_workers) {

  if (BatchSize > 1 && training) {
    int chunkSize = BatchSize / numThreads;
    int idx = 0;
    for (int i = 0; i < numThreads; i++) {
      idx = chunkSize * i;
      _workers[i]->doAsync(std::bind(
          &Layer::CalcInputsDelegate, this, prevLayerOutput, prevLayerSize,
          prevLayerOutputBatch, training, idx, idx + chunkSize));
    }
    for (int k = 0; k < numThreads; k++)
      _workers[k]->wait();
  } else {
    int chunkSize = Size / numThreads == 0 ? 1 : Size / numThreads;
    int iterator = numThreads > Size ? Size : numThreads;

    for (int i = iterator; i--;) {
      int start = i * chunkSize;
      int end = (i + 1) == iterator ? Size : (i + 1) * chunkSize;

      _workers[i]->doAsync(
          std::bind(&Layer::CalcInputsDelegate, this, prevLayerOutput,
                    prevLayerSize, prevLayerOutputBatch, training, start, end));
    }
    for (int i = iterator; i--;)
      _workers[i]->wait();
  }
}

//---------------------------------------------------
void Layer::CalculateOutputsThreaded(int &numThreads, bool &training,
                                     bool &countingRohat,
                                     std::vector<WorkerThread *> &_workers) {
  if (BatchSize > 1 && training) {
    int chunkSize = BatchSize / numThreads;
    int idx = 0;

    for (int i = numThreads; i--;) {
      idx = chunkSize * i;
      _workers[i]->doAsync(std::bind(&Layer::CalcOutputsDelegate, this, idx,
                                     idx + chunkSize, training, countingRohat));
    }
    for (int k = numThreads; k--;)
      _workers[k]->wait();
  } else {
    int chunkSize = Size / numThreads == 0 ? 1 : Size / numThreads;
    int iterator = numThreads > Size ? Size : numThreads;

    for (int i = iterator; i--;) {
      int start = i * chunkSize;
      int end = (i + 1) == iterator ? Size : (i + 1) * chunkSize;
      _workers[i]->doAsync(std::bind(&Layer::CalcOutputsDelegate, this, start,
                                     end, training, countingRohat));
    }
    for (int i = iterator; i--;)
      _workers[i]->wait();
  }
}

void Layer::CalcInputsDelegate(float *prevLayerOutput, int prevLayerSize,
                               float **prevLayerOutputBatch, bool &training,
                               int &start, int &end) {
  float result;
  int biasShift = UsingBias ? 1 : 0;
  if (BatchSize > 1 && training) {
    for (int batch = start; batch < end; batch++) {
      for (int k = 0; k < Size; k++) {
        result = 0.0;
        if (UsingBias && k == 0)
          continue;
        if (Mask[k] == 0.0f) {
          InputsBatch[batch][k] = 0.0f;
          continue;
        }
        const int w = (k - biasShift) * prevLayerSize;
        for (int i = 0; i < prevLayerSize; i++) {
          result += prevLayerOutputBatch[batch][i] * Weights[w + i];
        }
        InputsBatch[batch][k] = result;
      }
    }
  } else {
    for (int k = start; k < end; k++) {
      if (UsingBias && k == 0)
        continue;
      result = 0.0f;
      if (Mask[k] == 0.0f) {
        Inputs[k] = 0.0f;
        continue;
      }
      const int w = (k - biasShift) * prevLayerSize;
      for (int i = 0; i < prevLayerSize; i++) {
        result += prevLayerOutput[i] * Weights[w + i];
      }
      Inputs[k] = result;
    }
  }
}

//---------------------------------------------------
void Layer::CalcOutputsDelegate(int &start, int &end, bool &training,
                                bool &countingRohat) {
  // TODO ჩასამატებელია SoftMax რეალიზაცია
  if (BatchSize > 1 && training) {
    for (int batch = start; batch < end; batch++) {
      int vStart = 0, vEnd = Size;
      ActivateWith(InputsBatch[batch], OutputsBatch[batch], Mask, vStart, vEnd,
                   ActivationFunction);
      if (UsingBias)
        OutputsBatch[batch][0] = InputsBatch[batch][0];
    }
  } else {
    ActivateWith(Inputs, Outputs, Mask, start, end, ActivationFunction);
    if (countingRohat)
      for (int i = start; i < end; i++)
        RoHat[i] += Outputs[i];
    if (start == 0 && UsingBias)
      Outputs[0] = Inputs[0];
  }
}

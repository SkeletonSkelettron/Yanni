#include "neuralNetwork.h"
#include "../functions/statisticFunctions.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int pLS_ = 0;
int biasShift_ = 0;
int curLayerSize = 0;
void NeuralNetwork::NeuralNetworkInit() {
  iterations = 0;
  beta1Pow = 0.9;
  beta2Pow = 0.999;
  betaAELR = 0.001;
  ro = -0.9;
  for (int i = 0; i < ThreadCount; i++) {
    workers.push_back(new WorkerThread());
  }
  GradientsTemp = new float **[ThreadCount];
  lossesTmp = new float[ThreadCount > Layers[LayersSize - 1].Size
                            ? Layers[LayersSize - 1].Size
                            : ThreadCount];
  lambda = 0.7;
}

void NeuralNetwork::ShuffleDropoutsPlain() {
  for (int k = 1; k < LayersSize; k++) {
    Layer &L = Layers[k];
    if (L.DropOutSize <= 0.0f ||
        L.LayerType != NeuralEnums::LayerType::HiddenLayer)
      continue;

    const int biasShift = L.UsingBias ? 1 : 0;
    const int live = L.Size - biasShift;
    const int threads = ThreadCount > live ? live : ThreadCount;
    const int chunk = live / threads;

    for (int t = 0; t < threads; t++) {
      const int from = biasShift + t * chunk;
      const int to = (t + 1 == threads) ? L.Size : from + chunk;
      const int n = to - from;

      int dropPer = (int)(n * L.DropOutSize);
      if (dropPer >= n)
        dropPer = n - 1; // ერთი მაინც გადარჩეს
      const float scale = (float)n / (float)(n - dropPer);

      for (int i = from; i < to; i++)
        L.Mask[i] = scale;
      for (int d = 0; d < dropPer; d++) {
        int i;
        do {
          i = from + rand() % n;
        } while (L.Mask[i] == 0.0f);
        L.Mask[i] = 0.0f;
      }
    }
    if (biasShift)
      L.Mask[0] = 1.0f;
  }
}

void NeuralNetwork::InitializeWeights() {
  Layers[0].WeightsSize = 1;
  Layers[0].Weights = new float[1]{0};
  for (int i = 1; i < LayersSize; i++) {
    int size =
        Layers[i - 1].Size * (Layers[i].Size - (Layers[i].UsingBias ? 1 : 0));
    Layers[i].Weights = new float[size]{0};
    Layers[i].TempWeights = new float[size]{0};
    if (BatchSize > 1) {
      Layers[i].GradientsBatch = new float *[ThreadCount];
      for (int v = 0; v < ThreadCount; v++) {
        Layers[i].GradientsBatch[v] = new float[size]{0};
        for (int b = 0; b < size; b++)
          Layers[i].GradientsBatch[v][b] = 0;
      }
    } else
      Layers[i].Gradients = new float[size]{0};
    Layers[i].GradientsLR = new float[size]{0};
    Layers[i].Parameters = new float[size]{0};
    Layers[i].WeightsSize = size;
    Layers[i].MultipliedSums = new float[size]{0};

    for (int j = 0; j < size; j++)
      Layers[i].Weights[j] = Layers[i].UsingBias && j % Layers[i - 1].Size == 0
                                 ? 1.0f
                                 : (float)(rand() % 100);

    int minMax[2];
    bool sigmoidLike = Layers[i].ActivationFunction ==
                       NeuralEnums::ActivationFunction::Sigmoid;
    float start = sigmoidLike ? -1.0f : -0.07f;
    float end = sigmoidLike ? 1.0f : 0.07f;
    StandartizeLinearContract(Layers[i].Weights, size, minMax, start, end);

    // bias column carries a fixed 1.0, not a scaled random value
    if (Layers[i - 1].UsingBias)
      for (int j = 0; j < size; j++)
        if (j % Layers[i - 1].Size == 0)
          Layers[i].Weights[j] = 1.0f;
  }
}

float NeuralNetwork::PropagateForwardThreaded(bool training,
                                              bool countingRohat) {
  // if (training)
  // ShuffleDropoutsPlain();
  for (int k = 1; k < LayersSize - (countingRohat ? 1 : 0); k++) {
    Layers[k].CalculateInputsThreaded(Layers[k - 1].Outputs, Layers[k - 1].Size,
                                      Layers[k - 1].OutputsBatch, training,
                                      ThreadCount, workers);
    Layers[k].CalculateOutputsThreaded(ThreadCount, training, countingRohat,
                                       workers);
  }
  if (this->LossCalculation == NeuralEnums::LossCalculation::Full &&
      !countingRohat)
    return CalculateLoss(training);
  return -1;
}

void NeuralNetwork::PropagateBackThreaded() {
  // ClearNetwork();
  //  PopagateBackDelegateBatch2(0, 1, vector);
  if (LearningRateType == NeuralEnums::LearningRateType::Adam) {
    iterations++;
    // beta^t, not beta^(t+1): beta1Pow starts at 0.9, so multiplying before
    // first use overshoots by one power
    beta1Pow = pow(beta1, iterations);
    beta2Pow = pow(beta2, iterations);
  }
  // TODO ეს აქ არ უნდა იყოს
  if (BatchSize > 1) {
    int chunkSize = BatchSize / ThreadCount;
    int idx = 0;

    for (int i = 0; i < ThreadCount; i++) {
      idx = chunkSize * i;

      workers[i]->doAsync(std::bind(&NeuralNetwork::PropagateBackDelegateBatch,
                                    this, idx, idx + chunkSize, i));
    }
    for (int k = 0; k < ThreadCount; k++)
      workers[k]->wait();
    CalculateWeightsBatch();
  } else {
    for (unsigned int i = LayersSize - 1; i >= 1; i--) {
      pLS_ = Layers[i - 1].Size;
      biasShift_ = Layers[i].UsingBias ? 1 : 0;
      curLayerSize = Layers[i].Size;

      int Size = Layers[i].Size;
      int chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
      int threadsNum = ThreadCount > Size ? Size : ThreadCount;

      for (int threadId = 0; threadId < threadsNum; threadId++) {
        int start = threadId * chunkSize;
        int end =
            (threadId + 1) == threadsNum ? Size : (threadId + 1) * chunkSize;
        // PropagateBackDelegate ზე გადართვისას int Size =
        // Layers[i].IndexVectorSize; უნდა მივუთითო
        workers[threadId]->doAsync(std::bind(
            &NeuralNetwork::PropagateBackDelegate, this, i, start, end));
      }

      for (int k = 0; k < threadsNum; k++)
        workers[k]->wait();
    }
  }
}

// 66 წამი
void NeuralNetwork::PropagateBackDelegate(int i, int start, int end) {
  int numberIndex = 0;
  start = start < biasShift_ ? biasShift_ : start;
  for (int j = start; j < end; j++) {
    if (Layers[i].Mask[j] == 0.0f) {
      Layers[i].Inputs[j] = 0.0f;
      Layers[i].Outputs[j] = 0.0f;
      continue;
    }
    // Output ლეიერი
    if (i == LayersSize - 1)
      Layers[i].Outputs[j] =
          DifferentiateLossWith(Layers[i].Outputs[j], Layers[i].Target[j],
                                LossFunctionType, Layers[i].Size);
    else {
      int nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;
      Layers[i].Outputs[j] = 0;

      // l = 0 შემდეგი ლეიერის bias-ია: მას წონების მწკრივი არ აქვს და დელტაც
      for (int l = nextLayerBiasShift; l < Layers[i + 1].Size; l++) {
        Layers[i].Outputs[j] +=
            Layers[i + 1].Inputs[l] *
            Layers[i + 1]
                .TempWeights[(l - nextLayerBiasShift) * curLayerSize + j];
      }
    }
    Layers[i].Inputs[j] =
        Layers[i].Outputs[j] * Layers[i].Mask[j] *
        DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction,
                          Layers[i].Inputs);

    for (int p = 0; p < Layers[i - 1].Size; p++) {
      numberIndex = pLS_ * (j - biasShift_) + p;
      if (i != 1)
        Layers[i].TempWeights[numberIndex] = Layers[i].Weights[numberIndex];
      // Layers[i].Gradients[numberIndex] = ... if gradient optimization is
      // needed
      Layers[i].Weights[numberIndex] -=
          Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] *
          LearningRate; // Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] ეს
                        // არის გრადიენტი
                        // GetLearningRateMultipliedByGrad(gradient/*Layers[i].Gradients[numberIndex]*/,
                        // i, numberIndex);
    }
  }
}
/*
void NeuralNetwork::PropagateBackDelegateNew(int i, int start, int end) {
  int numberIndex = 0;
  int pLS = Layers[i - 1].Size;
  int biasShift = Layers[i].UsingBias ? 1 : 0;
  float gradient;
  int j = 0, p = 0, n = 0;
  float gradientTemp;
  for (int jj = start; jj < end; jj++) {
    j = Layers[i].IndexVector[jj];
    // Output ლეიერი
    if (i == LayersSize - 1) {
      Layers[i].Outputs[j] =
          DifferentiateLossWith(Layers[i].Outputs[j], Layers[i].Target[j],
                                LossFunctionType, Layers[i].Size);

      Layers[i].Inputs[j] =
          DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction,
                            Layers[i].Inputs, Layers[i].Mask);

      // Output ლეიერში წონების დაკორექტირება
      for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++) {
        p = Layers[i - 1].IndexVectorForNextLayer[pp];
        numberIndex = pLS * (j - biasShift) + p;
        Layers[i].MultipliedSums[numberIndex] = Layers[i].Weights[numberIndex] *
                                                Layers[i].Inputs[j] *
                                                Layers[i].Outputs[j];
        // gradient = Layers[i].Outputs[j] * Layers[i].Inputs[j] * Layers[i -
        // 1].Outputs[p];

        Layers[i].Weights[numberIndex] -=
            Layers[i].Outputs[j] * Layers[i].Inputs[j] *
            Layers[i - 1].Outputs[p] *
            LearningRate; // GetLearningRateMultipliedByGrad(gradient,
                          // i, numberIndex);
      }
    } else {
      Layers[i].Inputs[j] =
          DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction,
                            Layers[i].Inputs, Layers[i].Mask);

      float sum = 0;
      int nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;

      // შემდეგი ლეიერიდან უნდა აიღოს შესაბამისი ჯამები
      for (long int n = Layers[i + 1].UsingBias ? 1 : 0; n < Layers[i + 1].Size;
           n++) {
        sum +=
            Layers[i + 1]
                .MultipliedSums[(n - nextLayerBiasShift) * Layers[i].Size + j];
      }

      // მიმდინარე ნეირონის შესაბამისი წონების განახლება

      float mult = Layers[i].Inputs[j] * sum;

      if (i != 1)
        for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++) {
          // mult * Layers[i - 1].Outputs[n] არის gradient
          n = Layers[i - 1].IndexVectorForNextLayer[pp];
          numberIndex = pLS * (j - biasShift) + n;
          Layers[i].MultipliedSums[numberIndex] =
              Layers[i].Weights[numberIndex] * mult;
          gradient = mult * Layers[i - 1].Outputs[n];

          Layers[i].Weights[numberIndex] -=
              mult * Layers[i - 1].Outputs[n] *
              LearningRate; // GetLearningRateMultipliedByGrad(gradient,
                            // i, numberIndex);
        }
      else
        for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++) {
          n = Layers[i - 1].IndexVectorForNextLayer[pp];
          numberIndex = pLS * (j - biasShift) + n;
          gradient = mult * Layers[i - 1].Outputs[n];
          Layers[i].Weights[numberIndex] -=
              mult * Layers[i - 1].Outputs[n] *
              LearningRate; // GetLearningRateMultipliedByGrad(gradient,
                            // i, numberIndex);
        }
    }
  }
}
*/
void NeuralNetwork::PropagateBackDelegateBatch(int start, int end,
                                               int threadNum) {
  int pLS = 0;
  int biasShift = 0;
  int MaxLayerSize = 0;
  for (int k = 0; k < LayersSize; k++)
    MaxLayerSize = std::max(MaxLayerSize, Layers[k].Size);
  std::vector<float> outputsTemp(MaxLayerSize);
  for (int batch = start; batch < end; batch++) {
    for (int i = LayersSize - 1; i >= 1; i--) {
      std::fill(outputsTemp.begin(), outputsTemp.begin() + Layers[i - 1].Size,
                0.0f);
      pLS = Layers[i - 1].Size;
      biasShift = Layers[i].UsingBias ? 1 : 0;

      for (int j = biasShift; j < Layers[i].Size; j++) {
        // j = Layers[i].IndexVector[jj];
        //  Output ლეიერი
        if (i == LayersSize - 1)
          Layers[i].OutputsBatch[batch][j] =
              DifferentiateLossWith(Layers[i].OutputsBatch[batch][j],
                                    Layers[i].TargetsBatch[batch][j],
                                    LossFunctionType, Layers[i].Size);

        if (Layers[i].Mask[j] == 0.0f) {
          Layers[i].InputsBatch[batch][j] = 0.0f;
          Layers[i].OutputsBatch[batch][j] = 0.0f;
          continue;
        }
        Layers[i].InputsBatch[batch][j] =
            Layers[i].OutputsBatch[batch][j] * Layers[i].Mask[j] *
            DifferentiateWith(Layers[i].InputsBatch[batch][j],
                              Layers[i].ActivationFunction,
                              Layers[i].InputsBatch[batch]);

        const int w = pLS * (j - biasShift);

        const float d = Layers[i].InputsBatch[batch][j];
        float *g = Layers[i].GradientsBatch[threadNum] + w;
        const float *po = Layers[i - 1].OutputsBatch[batch];

        if (i != 1) {
          const float *pw = Layers[i].Weights + w;
          for (int p = 0; p < pLS; p++) {
            outputsTemp[p] += d * pw[p];
            g[p] += d * po[p];
          }
        } else {
          for (int p = 0; p < pLS; p++)
            g[p] += d * po[p];
        }
        //
      }
      if (i != 1) // ამის ოპტიმიზაცია შეიძლება
        for (int p = /*Layers[i - 1].UsingBias ? 1 :*/ 0; p < pLS; p++) {
          if (Layers[i - 1].Mask[p] == 0.0f)
            continue;
          Layers[i - 1].OutputsBatch[batch][p] = outputsTemp[p];
        }
    }
  }
}

void NeuralNetwork::CalculateWeightsBatch() {
  for (unsigned int i = LayersSize - 1; i >= 1; i--) {

    int Size = Layers[i].Size;
    int chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
    int iterator = ThreadCount > Size ? Size : ThreadCount;

    for (int l = 0; l < iterator; l++) {

      int start = l * chunkSize;
      int end = (l + 1) == iterator ? Size : (l + 1) * chunkSize;
      workers[l]->doAsync(std::bind(&NeuralNetwork::CalculateWeightsBatchSub,
                                    this, i, start, end));
    }
    for (int k = 0; k < iterator; k++)
      workers[k]->wait();
  }
}
void NeuralNetwork::CalculateWeightsBatchSub(int i, int start, int end) {
  float gradient = 0;
  int numberIndex = 0;
  const int pLS = Layers[i - 1].Size;
  const int biasShift = Layers[i].UsingBias ? 1 : 0;
  start = start < biasShift ? biasShift : start;
  for (int j = start; j < end; j++) {
    // მასკის შემოწმება არ სჭირდება: გამორთულ ნეირონზე გრადიენტი ისედაც ნულია,
    // შემოწმების გარეშე კი ყველა უჯრა ზუსტად ერთხელ  ნულდება
    for (int p = 0; p < pLS; p++) {
      numberIndex = pLS * (j - biasShift) + p;
      for (int t = 0; t < ThreadCount; t++) {
        gradient += Layers[i].GradientsBatch[t][numberIndex];
        Layers[i].GradientsBatch[t][numberIndex] = 0;
      }
      gradient /= BatchSize;
      Layers[i].Weights[numberIndex] -=
          GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
      gradient = 0;
    }
  }
  gradient = 0;
}

// dropout-ის გამორთვა: ტესტირებისას ყველა ნეირონი მონაწილეობს, სკალირების
// გარეშე
void NeuralNetwork::PrepareForTesting() {
  for (int k = 0; k < LayersSize; k++)
    std::fill(Layers[k].Mask, Layers[k].Mask + Layers[k].Size, 1.0f);
}

float NeuralNetwork::CalculateLoss(bool &training) {
  float *losses;
  if (BatchSize > 1 && training) {
    int chunkSize = BatchSize / ThreadCount;
    int idx = 0;

    losses = new float[ThreadCount];
    for (int h = 0; h < ThreadCount; h++) {
      losses[h] = 0;
    }
    float result;
    for (int i = 0; i < ThreadCount; i++) {
      idx = chunkSize * i;

      workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossBatchSub, this,
                                    idx, idx + chunkSize, std::ref(losses[i])));
    }
    for (int k = 0; k < ThreadCount; k++)
      workers[k]->wait();
    float sum = (float)0.0;
    for (int h = 0; h < ThreadCount; h++) {
      sum += losses[h];
    }
    delete[] (losses);
    return sum;
  }
  if (BatchSize == 0 || training) {
    int chunkSize = Layers[LayersSize - 1].Size / ThreadCount == 0
                        ? 1
                        : Layers[LayersSize - 1].Size / ThreadCount;
    int hidenchunkSize = 0;
    if (Type == NeuralEnums::NetworkType::AutoEncoder &&
        AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce) {
      int hidenchunkSize = Layers[LayersSize - 2].Size / ThreadCount == 0
                               ? 1
                               : Layers[LayersSize - 2].Size / ThreadCount;
    }
    int iterator = ThreadCount > Layers[LayersSize - 1].Size
                       ? Layers[LayersSize - 1].Size
                       : ThreadCount;

    for (size_t i = 0; i < (ThreadCount > Layers[LayersSize - 1].Size
                                ? Layers[LayersSize - 1].Size
                                : ThreadCount);
         i++) {
      lossesTmp[i] = 0;
    };

    for (int i = 0; i < iterator; i++) {
      int start = i * chunkSize;
      int end = (i + 1) == iterator ? Layers[LayersSize - 1].Size
                                    : (i + 1) * chunkSize;
      int klbstart = i * hidenchunkSize;
      int klbend = (i + 1) == iterator ? Layers[LayersSize - 2].Size
                                       : (i + 1) * hidenchunkSize;
      workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossSub, this,
                                    start, end, klbstart, klbend,
                                    std::ref(lossesTmp[i])));
    }
    for (int k = 0; k < iterator; k++)
      workers[k]->wait();

    float result = 0.0;
    for (size_t i = 0; i < (ThreadCount > Layers[LayersSize - 1].Size
                                ? Layers[LayersSize - 1].Size
                                : ThreadCount);
         i++)
      result += lossesTmp[i];

    float regularizerCost = 0.0;
    if (Type == NeuralEnums::NetworkType::AutoEncoder &&
        AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce) {
      for (size_t w = 1; w < LayersSize; w++) {
        for (size_t y = 0; y < (Layers[w].UsingBias
                                    ? (Layers[w].Size - 1) * Layers[w - 1].Size
                                    : Layers[w].Size * Layers[w - 1].Size);
             y++) {
          regularizerCost += Layers[w].Weights[y] * Layers[w].Weights[y];
        }
      }
    }
    return result + regularizerCost * lambda / 2.0;
  }
}

void NeuralNetwork::CalculateLossBatchSub(int start, int end, float &loss) {
  float result = 0.0;
  for (size_t i = start; i < end; i++) {
    result += CalculateLossFunction(
        LossFunctionType, Layers[LayersSize - 1].OutputsBatch[i],
        Layers[LayersSize - 1].TargetsBatch[i], 0, Layers[LayersSize - 1].Size,
        Layers[LayersSize - 1].Size);
  }
  loss = result;
}

void NeuralNetwork::CalculateLossSub(int start, int end, int klbstart,
                                     int klbend, float &loss) {
  float result = 0.0;
  float klbResult = 0.0;

  // CalculateLossFunction already sums over [start, end), so no outer loop
  // needed
  result = CalculateLossFunction(
      LossFunctionType, Layers[LayersSize - 1].Outputs,
      Layers[LayersSize - 1].Target, start, end, Layers[LayersSize - 1].Size);

  if (Type == NeuralEnums::NetworkType::AutoEncoder &&
      AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce) {
    klbResult +=
        KullbackLeiblerDivergence(Layers[1].RoHat, ro, klbstart, klbend);
  }
  loss = result + klbResult;
}

float NeuralNetwork::GetLearningRateMultipliedByGrad(float &gradient,
                                                     int &iterator, int &j) {
  switch (LearningRateType) {
  case NeuralEnums::LearningRateType::Static: {
    return LearningRate * gradient;
    break;
  }
  case NeuralEnums::LearningRateType::AdaGrad: {
    return AdaGrad(Layers[iterator].GradientsLR, gradient, j) * gradient;
    break;
  }
  case NeuralEnums::LearningRateType::AdaDelta: {
    return AdaDelta(Layers[iterator].GradientsLR, Layers[iterator].Parameters,
                    gradient, j) *
           gradient;
    break;
  }
  // following 3 methods does not require gradient multiplication
  case NeuralEnums::LearningRateType::Adam: {
    return Adam(gradient, j, iterator);
    break;
  }
  case NeuralEnums::LearningRateType::AdaMax: {
    return AdaMax(gradient, j, iterator);
    break;
  }
  case NeuralEnums::LearningRateType::AdamMod: {
    return AdamMod(gradient, j, iterator);
    break;
  }
  case NeuralEnums::LearningRateType::RMSProp: {
    return RMSProp(Layers[iterator].GradientsLR, gradient, j) * gradient;
    break;
  }
  default: {
    throw std::runtime_error("learning rate function not defined");
    break;
  }
  }
}

float NeuralNetwork::Adam(float &gradient, int &j, int &iterator) {
  float result, param;

  // mt
  Layers[iterator].Parameters[j] =
      beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
  // vt
  Layers[iterator].GradientsLR[j] = beta2 * Layers[iterator].GradientsLR[j] +
                                    (1 - beta2) * gradient * gradient;

  // bias-corrected estimates: mHat = m/(1-beta1^t), vHat = v/(1-beta2^t)
  // standard Adam: lr * mHat / (sqrt(vHat) + epsilon)
  return (LearningRate * Layers[iterator].Parameters[j]) /
         ((1 - beta1Pow) *
          (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2Pow)) + epsilon));
}

float NeuralNetwork::AdaGrad(float *gradients, float &gradient, int &j) {
  gradients[j] += gradient * gradient;
  return 0.01 / sqrt(gradients[j] + epsilon);
}

float NeuralNetwork::AdaDelta(float *gradients, float *parameters,
                              float &Gradient, int &j) {
  float result, param;
  gradients[j] = momentum * gradients[j] + (1 - momentum) * Gradient * Gradient;
  result = sqrt(parameters[j] + epsilon) / sqrt(gradients[j] + epsilon);
  param = result * Gradient;
  parameters[j] = momentum * parameters[j] + (1 - momentum) * param * param;
  return result;
}

float NeuralNetwork::AdamMod(float &Gradient, int &j, int &iterator) {
  float result, param;
  float prelim = (1 - momentum) * Gradient;

  Layers[iterator].GradientsLR[j] =
      momentum * Layers[iterator].GradientsLR[j] + prelim * Gradient;
  Layers[iterator].Parameters[j] =
      momentum * Layers[iterator].Parameters[j] + prelim;

  return (LearningRate * Layers[iterator].Parameters[j] / (1 - beta1Pow)) /
         (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2Pow)) + epsilon);
}

float NeuralNetwork::AdaMax(float &gradient, int &j, int &iterator) {
  float result, param;

  // mt
  Layers[iterator].Parameters[j] =
      beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
  // vt
  Layers[iterator].GradientsLR[j] =
      std::max(beta2 * Layers[iterator].GradientsLR[j], abs(gradient));

  return (LearningRate * Layers[iterator].Parameters[j]) /
         ((1 - beta1Pow) * Layers[iterator].GradientsLR[j]);
}

float NeuralNetwork::RMSProp(float *gradients, float &gradient, int &j) {
  gradients[j] = momentum * gradients[j] + (1 - momentum) * gradient * gradient;
  return startingLearningRate / sqrt(gradients[j] + epsilon);
}

static const char *ActName(NeuralEnums::ActivationFunction a) {
  using A = NeuralEnums::ActivationFunction;
  switch (a) {
  case A::None:
    return "None";
  case A::Sigmoid:
    return "Sigmoid";
  case A::Tanh:
    return "Tanh";
  case A::ReLU:
    return "ReLU";
  case A::MReLU:
    return "MReLU";
  case A::SoftMax:
    return "SoftMax";
  case A::GeLU:
    return "GeLU";
  case A::SoftPlus:
    return "SoftPlus";
  case A::SoftSign:
    return "SoftSign";
  }
  return "?";
}
static const char *LayerName(NeuralEnums::LayerType t) {
  using L = NeuralEnums::LayerType;
  switch (t) {
  case L::InputLayer:
    return "Input";
  case L::HiddenLayer:
    return "Hidden";
  case L::OutputLayer:
    return "Output";
  case L::None:
    return "None";
  }
  return "?";
}
static const char *LrName(NeuralEnums::LearningRateType t) {
  using R = NeuralEnums::LearningRateType;
  switch (t) {
  case R::Static:
    return "Static";
  case R::Adam:
    return "Adam";
  case R::AdaGrad:
    return "AdaGrad";
  case R::AdaDelta:
    return "AdaDelta";
  case R::AdamMod:
    return "AdamMod";
  case R::AdaMax:
    return "AdaMax";
  case R::AMSGrad:
    return "AMSGrad";
  case R::Cyclic:
    return "Cyclic";
  case R::GuraMethod:
    return "GuraMethod";
  case R::Nadam:
    return "Nadam";
  case R::RMSProp:
    return "RMSProp";
  }
  return "?";
}

void PrintNetworkInfo(NeuralNetwork &nn, size_t trainingSetSize) {
  printf("\n=== network ===\n");
  printf("  batch %d | threads %d | lr %.4g (%s)\n", nn.BatchSize,
         nn.ThreadCount, nn.LearningRate, LrName(nn.LearningRateType));

  printf("\n  %-3s %-7s %-9s %5s %4s  %-13s %10s\n", "#", "type", "act", "size",
         "bias", "weights", "params");
  long total = 0;
  for (int i = 0; i < nn.LayersSize; i++) {
    Layer &L = nn.Layers[i];
    int bs = L.UsingBias ? 1 : 0;
    long w = (i == 0) ? 0 : (long)nn.Layers[i - 1].Size * (L.Size - bs);
    total += w;
    char shape[32] = "-";
    if (i)
      snprintf(shape, sizeof shape, "%d x %d", L.Size - bs,
               nn.Layers[i - 1].Size);
    printf("  %-3d %-7s %-9s %5d %4s  %-13s %10ld\n", i, LayerName(L.LayerType),
           ActName(L.ActivationFunction), L.Size, L.UsingBias ? "yes" : "no",
           shape, w);
  }
  printf("  %54s %10ld\n", "total parameters:", total);
  printf("  %54s %9.1f MB\n", "weights memory:", total * sizeof(float) / 1e6);

  // things that can be silently wrong
  printf("\n  checks:\n");
  if (nn.BatchSize > 1 && nn.BatchSize % nn.ThreadCount)
    printf("    !! BatchSize %% ThreadCount = %d -> %d samples per batch are\n"
           "       silently skipped, but the gradient is still divided by "
           "BatchSize\n",
           nn.BatchSize % nn.ThreadCount, nn.BatchSize % nn.ThreadCount);
  else
    printf("    ok  BatchSize divides evenly by ThreadCount\n");

  if (nn.BatchSize > 0 && trainingSetSize)
    printf("    %zu samples / batch %d = %zu weight updates per epoch\n",
           trainingSetSize, nn.BatchSize, trainingSetSize / nn.BatchSize);

  for (int i = 1; i < nn.LayersSize; i++)
    if (nn.Layers[i].WeightsSize == 0)
      printf("    !! layer %d: WeightsSize = 0 (the CUDA path sizes its\n"
             "       cudaMalloc from this)\n",
             i);

  // Everything below is read straight out of the mask, so it reports what the
  // network will actually do rather than what the config asked for.
  for (int i = 0; i < nn.LayersSize; i++) {
    Layer &L = nn.Layers[i];
    if (!L.Mask)
      continue;
    int biasShift = L.UsingBias ? 1 : 0;
    int neurons = L.Size - biasShift;
    long dropped = std::count(L.Mask + biasShift, L.Mask + L.Size, 0.0f);
    if (dropped == 0)
      continue;

    int kept = neurons - (int)dropped;
    float rate = (float)dropped / (float)neurons;
    float expected = kept > 0 ? (float)neurons / (float)kept : 0.0f;

    // the scale the mask actually carries, taken from the first live neuron
    float stored = 0.0f;
    for (int k = biasShift; k < L.Size; k++)
      if (L.Mask[k] != 0.0f) {
        stored = L.Mask[k];
        break;
      }

    printf("    layer %d: %ld of %d dropped (p=%.2f), scale %.4f", i, dropped,
           neurons, rate, stored);
    if (kept == 0)
      printf("   !! every neuron dropped\n");
    else if (std::fabs(stored - expected) > 1e-3f)
      printf("   !! expected 1/(1-p) = %.4f\n", expected);
    else
      printf("   ok\n");

    if (L.UsingBias && L.Mask[0] == 0.0f)
      printf("    !! layer %d: the bias neuron is masked out\n", i);
  }
  printf("\n");
}

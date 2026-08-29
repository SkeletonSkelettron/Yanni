#include "activationFunctions.h"
#include "../core/enums.h"
#include <cmath>
#include <math.h>
#include <stdexcept>

const float PI2 = 6.2831853071795f;
const float SQ2 = 1.4142135623730f;

// void BalanceWith(std::vector <float>& dataset, NeuralEnums::BalanceType
// BalancingMethod)
//{
//	switch (BalancingMethod)
//	{
//	case NeuralEnums::BalanceType::None: break;
//	case NeuralEnums::BalanceType::GaussianStandartization:
//	{
//		Standartize(dataset, dataset.size());
//		break;
//	}
//	case NeuralEnums::BalanceType::Normalization:
//	{
//		std::vector<int> tmp;
//		tmp.resize(2);
//		Compress(dataset,dataset.size() , tmp);
//		break;
//	}
//
//	default:
//		break;
//	}
// }

void ActivateWith(float *inputs, float *outputs, float *mask, int &start,
                  int &end, NeuralEnums::ActivationFunction &function) {
  switch (function) {
  case (NeuralEnums::ActivationFunction::Sigmoid): {
    Sigmoid_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::ReLU): {
    ReLU_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::MReLU): {
    MReLU_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::Tanh): {
    Tanh_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::GeLU): {
    GeLU_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::SoftPlus): {
    SoftPlus_v(inputs, outputs, mask, start, end);
    break;
  }
  case (NeuralEnums::ActivationFunction::SoftSign): {
    SoftSign_v(inputs, outputs, mask, start, end);
    break;
  }
  default:
    throw std::runtime_error("ActivationFunction not assigned");
  }
}

inline void GeLU_v(float *inputs, float *outputs, float *mask, int &start,
                   int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * GeLU(inputs[i]);
  }
}
inline void Sigmoid_v(float *inputs, float *outputs, float *mask, int &start,
                      int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * Sigmoid(inputs[i]);
  }
}
inline void Tanh_v(float *inputs, float *outputs, float *mask, int &start,
                   int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * tanh(inputs[i]);
  }
}
inline void MReLU_v(float *inputs, float *outputs, float *mask, int &start,
                    int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * MReLU(inputs[i]);
  }
}
inline void ReLU_v(float *inputs, float *outputs, float *mask, int &start,
                   int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * ReLU(inputs[i]);
  }
}
inline void SoftMax_v(float *inputs, float *inputsSoftMax, float *outputs,
                      float *mask, int &start, int &end) {
  // TODO მაინც კაი სანახავია როგორ მუშაობს
  // for (int i = 0; i < maskSize; i++);
  //  outputs[maskSize[i]] = SoftMax(inputs[indexVectorSize[i]],
  //  inputsSoftMax, dropoutNeurons);
}
inline void SoftPlus_v(float *inputs, float *outputs, float *mask, int &start,
                       int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * SoftPlus(inputs[i]);
  }
}
inline void SoftSign_v(float *inputs, float *outputs, float *mask, int &start,
                       int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * SoftSign(inputs[i]);
  }
}

inline void Assign_v(float *inputs, float *outputs, float *mask, int &start,
                     int &end) {
  for (int i = start; i < end; i++) {
    outputs[i] = mask[i] == 0.0f ? 0.0f : mask[i] * inputs[i];
  }
}

inline float SoftSign(float &x) { return x / (abs(x) + 1); }
inline float SoftSignDerivative(float &x) { return 1.0 / pow(1.0 + abs(x), 2); }

float SoftPlus(float &x) { return log(1.0 + exp(x)); }
inline float SoftPlusDerivative(float &x) { return 1.0 / (1.0 + exp(-x)); }
inline float SoftMax(float &x, float *layerInputs, float *mask, int &size) {
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += mask[i] == 0.0f ? 0.0 : exp(layerInputs[i]);
  }
  return exp(x) / sum;
}

inline float SoftMaxDerivative(float &x, float *inputs, float *mask,
                               int &size) {
  float y = SoftMax(x, inputs, mask, size);
  return y * (1.0 - y);
}

inline float Sigmoid(float &x) { return 1.0f / (1.0f + expf(-x)); }

inline float SigmoidDerivative(float &x) {
  float sigm = Sigmoid(x);
  return sigm * (1.0 - sigm);
}

inline float ReLU(float &x) { return x <= 0.0 ? 0.0 : x; }

inline float ReLUDerivative(float &x) { return x <= 0.0 ? 0.0 : 1.0; }

inline float Tanh(float &x) { return tanh(x); }

inline float TanhDerivative(float &x) { return 1.0 - tanh(x) * tanh(x); }

inline float MReLU(float &x) { return x < 0.0 ? 0.0005 * x : x; }

inline float MReLUDerivative(float &x) { return x < 0.0 ? 0.0005 : 1.0; }

inline float GeLU(float &x) { return 0.5 * x * (1.0 + erf(x / SQ2)); }

inline float GeLUDerivative(float &x) {
  return 0.5 * (1.0 + erf(x / SQ2)) + x * exp(-(x * x) / 2.0) / pow(PI2, 0.5);
}

int GetMaxIndex(float *outPut, int outpSize) {
  if (outpSize <= 0)
    throw std::runtime_error("GetMaxIndex: empty output");
  int index = 0;
  float val = outPut[0];
  for (int i = 0; i < outpSize; i++) {
    if (std::isnan(outPut[i]))
      throw std::runtime_error("GetMaxIndex: NaN in output layer");
    if (outPut[i] > val) {
      val = outPut[i];
      index = i;
    }
  }
  return index;
}
//

float exp1024(float x) {
  x = 1.0 + x / 256.0;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}

float DifferentiateWith(float &x, NeuralEnums::ActivationFunction &function,
                        float *inputs) {
  switch (function) {
  case (NeuralEnums::ActivationFunction::Sigmoid): {
    return SigmoidDerivative(x);
    break;
  }
  case (NeuralEnums::ActivationFunction::ReLU): {
    return ReLUDerivative(x);
    break;
  }
  case (NeuralEnums::ActivationFunction::MReLU): {
    return MReLUDerivative(x);
    break;
  }
  case (NeuralEnums::ActivationFunction::Tanh): {
    return TanhDerivative(x);
    break;
  }
  case (NeuralEnums::ActivationFunction::GeLU): {
    return GeLUDerivative(x);
    break;
  }
  case (NeuralEnums::ActivationFunction::SoftPlus): {
    return SoftPlusDerivative(x);
    break;
  }

  case (NeuralEnums::ActivationFunction::SoftSign): {
    return SoftSignDerivative(x);
    break;
  }
  default: {
    throw std::runtime_error("ActivationFunction not assigned");
    break;
  }
  }
}

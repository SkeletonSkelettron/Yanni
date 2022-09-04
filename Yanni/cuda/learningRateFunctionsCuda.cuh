//#include <math.h>
//#include "Enums.h"
//namespace NeuNetCuda {
//	class NeuralNetworkCuda;
//	const float momentum = 0.9;
//	const float epsilon = 0.0000001;
//	const float startingLearningRate = 0.001;
//	const float beta1 = 0.9;
//	const float beta2 = 0.999;
//
//	float GetLearningRateMultipliedByGradC(NeuralNetworkCuda& nn, float& gradient, int& iterator, int& j)
//	{
//		//if (nn.LearningRateType == NeuralEnums::LearningRateType::Static)
//		//	return nn.LearningRate * gradient;
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaGrad)
//		//	return AdaGrad(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaDelta)
//		//	return AdaDelta(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].Parameters, gradient, j) * gradient;
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::Adam)
//		//	return Adam(nn, gradient, j, iterator);
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaMax)
//		//	return AdaMax(nn, gradient, j, iterator);
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdamMod)
//		//	return AdamMod(nn, gradient, j, iterator);
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::RMSProp)
//		//	return RMSProp(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
//		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::GuraMethod)
//		//	return GuraMethod(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].LearningRates, gradient, j, nn.LearningRate) * gradient;
//		switch (nn.LearningRateType)
//		{
//		case NeuralEnums::LearningRateType::Static:
//			return nn.LearningRate * gradient;
//		case NeuralEnums::LearningRateType::AdaGrad:
//			return AdaGrad(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
//			//case NeuralEnums::LearningRateType::AdaDelta: 
//			//	return AdaDelta(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].Parameters, gradient, j) * gradient;
//			//	//following 3 methods does not require gradient multiplication
//			//case NeuralEnums::LearningRateType::Adam: 
//			//	return Adam(nn, gradient, j, iterator);
//			//case NeuralEnums::LearningRateType::AdaMax: 
//			//	return AdaMax(nn, gradient, j, iterator);
//			//case NeuralEnums::LearningRateType::AdamMod: 
//			//	return AdamMod(nn, gradient, j, iterator);
//			//case NeuralEnums::LearningRateType::RMSProp: 
//			//	return RMSProp(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
//			//case NeuralEnums::LearningRateType::GuraMethod: 
//			//	return GuraMethod(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].LearningRates, gradient, j, nn.LearningRate) * gradient;
//		default:
//		{
//
//		}
//		}
//	}
//
//
//	//float  Adam(NeuralNetworkCuda nn, float& gradient, int& j, int& iterator)
//	//{
//	//	float result, param;
//
//	//	//mt
//	//	nn.Layers[iterator].Parameters[j] = beta1 * nn.Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
//	//	//vt
//	//	nn.Layers[iterator].GradientsLR[j] = beta2 * nn.Layers[iterator].GradientsLR[j] + (1 - beta2) * gradient * gradient;
//
//
//	//	return (nn.LearningRate * nn.Layers[iterator].Parameters[j]) / ((1 - nn.beta1Pow) * (sqrt(nn.Layers[iterator].GradientsLR[j] / (1 - nn.beta2Pow)) + epsilon));
//	//}
//
//	//float AdaGrad(float* gradients, float& gradient, int& j)
//	//{
//	//	gradients[j] += gradient * gradient;
//	//	return 0.01 / sqrt(gradients[j] + epsilon);
//	//}
//
//
//	//float AdaDelta(float* gradients, float* parameters, float& Gradient, int& j)
//	//{
//	//	float result, param;
//	//	gradients[j] = momentum * gradients[j] + (1 - momentum) * Gradient * Gradient;
//	//	result = sqrt(parameters[j] + epsilon) / sqrt(gradients[j] + epsilon);
//	//	param = result * Gradient;
//	//	parameters[j] = momentum * parameters[j] + (1 - momentum) * param * param;
//	//	return result;
//	//}
//
//	//float AdamMod(NeuralNetworkCuda& nn, float& Gradient, int& j, int& iterator)
//	//{
//	//	float result, param;
//	//	float prelim = (1 - momentum) * Gradient;
//
//	//	nn.Layers[iterator].GradientsLR[j] = momentum * nn.Layers[iterator].GradientsLR[j] + prelim * Gradient;
//	//	nn.Layers[iterator].Parameters[j] = momentum * nn.Layers[iterator].Parameters[j] + prelim;
//
//	//	return (nn.LearningRate * nn.Layers[iterator].Parameters[j] / (1 - beta1)) / (sqrt(nn.Layers[iterator].GradientsLR[j] / (1 - beta2)) + epsilon);
//	//}
//
//
//	//float AdaMax(NeuralNetworkCuda& nn, float& gradient, int& j, int& iterator)
//	//{
//	//	float result, param;
//
//	//	//mt
//	//	nn.Layers[iterator].Parameters[j] = beta1 * nn.Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
//	//	//vt
//	//	nn.Layers[iterator].GradientsLR[j] = max(beta2 * nn.Layers[iterator].GradientsLR[j], abs(gradient));
//
//
//	//	return (nn.LearningRate * nn.Layers[iterator].Parameters[j]) / ((1 - nn.beta1Pow) * nn.Layers[iterator].GradientsLR[j]);
//	//}
//
//	//float RMSProp(float* gradients, float& gradient, int& j)
//	//{
//	//	gradients[j] = momentum * gradients[j] + (1 - momentum) * gradient * gradient;
//	//	return startingLearningRate / sqrt(gradients[j] + epsilon);
//	//}
//
//}
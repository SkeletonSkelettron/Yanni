#ifndef NEURALNETWORKCUDA_H
#define NEURALNETWORKCUDA_H
#include <math.h>
#include <string>
#include <condition_variable>
#include "enums.h"
#include "layerCuda.cuh"
#include "activationFunctionsCuda.cuh"
#include "lossFunctionsCuda.cuh"
//https://stackoverflow.com/questions/16350473/why-do-i-need-stdcondition-variable

__device__ struct NeuralNetworkCuda
{
	unsigned short ThreadCount;
	float LearningRate;
	int Type;
	int LearningRateType;
	int BalanceType;
	int LossFunctionType;
	int GradientType;
	int Metrics;
	int AutoEncoderType;
	int LossCalculation;
	int LogLoss;
	int BatchSize;
	bool Cuda;
	LayerCuda* Layers;
	int LayersSize;
	float*** GradientsTemp;
	float* lossesTmp;
	float ro;
	int iterations;
	float beta1Pow;
	float beta2Pow;
	float betaAELR;
	int MaxBachSize;
	const float momentum = 0.9;
	const float epsilon = 0.0000001;
	const float startingLearningRate = 0.001;
	const float beta1 = 0.9;
	const float beta2 = 0.999;

	int pLS_ = 0;
	int biasShift_ = 0;
	int curLayerSize = 0;
	//weight decay parameter for sparce autoencoder
	float lambda = 0.7;
	__host__ __device__ NeuralNetworkCuda() {}


	__device__ void ShuffleDropoutsPlain()
	{
		for (int k = 1; k < LayersSize; k++)
		{
			int biasShift = Layers[k].UsingBias ? 1 : 0;
			if (Layers[k].DropOutSize > 0 && Layers[k].LayerType == 1) // Hidden Layer
			{
				srand(time(NULL));
				int rnum = 0;
				bool tmp;
				for (int i = 0; i < Layers[k].Size; i++)
				{
					rnum = rand() % Layers[k].Size;
					tmp = Layers[k].DropoutNeurons[i];
					Layers[k].DropoutNeurons[i] = Layers[k].DropoutNeurons[rnum];
					Layers[k].DropoutNeurons[rnum] = tmp;
				}

				int counter = 0;
				if (biasShift == 1)
					Layers[k].IndexVectorForNextLayer[0] = 0;
				for (int i = biasShift; i < Layers[k].Size; i++)
					if (!Layers[k].DropoutNeurons[i])
					{
						Layers[k].IndexVector[counter] = i;
						Layers[k].IndexVectorForNextLayer[counter + biasShift] = i;
						counter++;
					}
			}
		}
	}

	__device__ void PropagateBack(int i, int batch, int start, int end) // i- layer index
	{
		int numberIndex = 0;
		float gradient;
		int nls = Layers[i + 1].Size;
		int j = 0, p = 0, l = 0;
		for (int jj = start; jj < end; jj++)
		{
			j = Layers[i].IndexVector[jj];
			// Output ლეიერი
			if (i == LayersSize - 1)
				Layers[i].GetOutputsBatch(batch, j) = DifferentiateLossWithCuda(Layers[i].GetOutputsBatch(batch, j), Layers[i].GetTargetsBatch(batch, j), LossFunctionType, Layers[i].Size);
			else
			{
				int nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;
				Layers[i].GetOutputsBatch(batch, j) = 0;

				for (int ll = Layers[i + 1].IndexVectorSize; ll--;)
				{
					l = Layers[i + 1].IndexVector[ll];
					Layers[i].GetOutputsBatch(batch, j) += Layers[i + 1].GetInputsBatch(batch, l) * Layers[i + 1].TempWeights[(l - nextLayerBiasShift) * curLayerSize + j];
				}
			}
			Layers[i].GetInputsBatch(batch, j) = Layers[i].GetOutputsBatch(batch, j) * DifferentiateWithCuda(Layers[i].GetInputsBatch(batch, j), Layers[i].ActivationFunction, Layers[i].GetInputsBatch(batch), Layers[i].DropoutNeurons);

			for (int pp = Layers[i - 1].IndexVectorForNextLayerSize; pp--;)
			{
				p = Layers[i - 1].IndexVectorForNextLayer[pp];
				numberIndex = pLS_ * (j - biasShift_) + p;
				if (i != 1)
					Layers[i].TempWeights[numberIndex] = Layers[i].Weights[numberIndex];
				if (BatchSize == 1)
					//Layers[i].Gradients[numberIndex] = Layers[i].Inputs[j] * Layers[i - 1].Outputs[p];// ... if gradient optimization is needed
					Layers[i].Weights[numberIndex] -= Layers[i].GetInputsBatch(batch, j) * Layers[i - 1].GetOutputsBatch(batch, p) * LearningRate;// Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] ეს არის გრადიენტი GetLearningRateMultipliedByGrad(gradient/*Layers[i].Gradients[numberIndex]*/, i, numberIndex);
			}
		}
	}
	__device__ void PropagateBackCalculateGradients(int i, int batchCount, int weights_start, int weights_end) // i- layer index start და end წონის ვექტორის ქაუნთერია;
	{
		int numberIndex = 0;
		float gradient = 0;
		int pls = Layers[i - 1].Size;
		int cli = 0, pli = 0;
		for (int wi = weights_start; wi < weights_end; wi++)
		{
			cli = Layers[i].IndexVector[wi / Layers[i - 1].IndexVectorForNextLayerSize];
			pli = Layers[i - 1].IndexVectorForNextLayer[wi % Layers[i - 1].IndexVectorForNextLayerSize];
			for (size_t b = 0; b < batchCount; b++)
			{
				gradient += Layers[i].GetInputsBatch(b, cli) * Layers[i - 1].GetOutputsBatch(b, pli);
			}
			gradient /= batchCount;
			int idx = pLS_ * (cli - biasShift_) + Layers[i - 1].IndexVectorForNextLayer[wi % Layers[i - 1].IndexVectorForNextLayerSize];
			Layers[i].Weights[idx] -= gradient * LearningRate;
			gradient = 0;
		}
	}

	__device__ void  StartTesting()
	{
		for (int k = 0; k < LayersSize; k++)
		{
			if (Layers[k].LayerType == static_cast<int>(NeuralEnums::LayerType::HiddenLayer) && Layers[k].DropOutSize > 0)
			{
				int dropoutCounter = 0;
				for (int i = 0; i < Layers[k].Size; i++)
				{
					if (Layers[k].DropoutNeurons[i])
						dropoutCounter++;
				}
				delete[](Layers[k].IndexVector);
				Layers[k].IndexVector = new int[dropoutCounter];

				int biasShift = Layers[k].UsingBias ? 1 : 0;
				int counter = 0;
				for (int i = biasShift; i < Layers[k].Size; i++)
				{
					if (Layers[k].DropoutNeurons[i])
					{
						Layers[k].IndexVector[counter] = i;
						counter++;
					}
				}
			}
		}
	}

	__device__ float CalculateLoss(bool& training)
	{
		float* losses;
		if (BatchSize > 1 && training)
		{
			int chunkSize = BatchSize / ThreadCount;
			int idx = 0;

			losses = new float[ThreadCount];
			for (int h = 0; h < ThreadCount; h++)
			{
				losses[h] = 0;
			}
			float result = 0;
			for (int i = 0; i < ThreadCount; i++)
			{
				idx = chunkSize * i;

				// workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossBatchSub, this, idx, idx + chunkSize, std::ref(losses[i])));
			}
			for (int k = 0; k < ThreadCount; k++)
				;//workers[k]->wait();
			float sum = (float)0.0;
			for (int h = 0; h < ThreadCount; h++)
			{
				sum += losses[h];
			}
			delete[](losses);
			return sum;
		}
		if (BatchSize == 0 || training)
		{
			int chunkSize = Layers[LayersSize - 1].Size / ThreadCount == 0 ? 1 : Layers[LayersSize - 1].Size / ThreadCount;
			int hidenchunkSize = 0;
			if (Type == static_cast<int>(NeuralEnums::NetworkType::AutoEncoder) && AutoEncoderType == static_cast<int>(NeuralEnums::AutoEncoderType::Sparce))
			{
				int hidenchunkSize = Layers[LayersSize - 2].Size / ThreadCount == 0 ? 1 : Layers[LayersSize - 2].Size / ThreadCount;
			}
			int iterator = ThreadCount > Layers[LayersSize - 1].Size ? Layers[LayersSize - 1].Size : ThreadCount;

			for (size_t i = 0; i < (ThreadCount > Layers[LayersSize - 1].Size ? Layers[LayersSize - 1].Size : ThreadCount); i++)
			{
				lossesTmp[i] = 0;
			};

			for (int i = 0; i < iterator; i++)
			{
				int start = i * chunkSize;
				int end = (i + 1) == iterator ? Layers[LayersSize - 1].Size : (i + 1) * chunkSize;
				int klbstart = i * hidenchunkSize;
				int klbend = (i + 1) == iterator ? Layers[LayersSize - 2].Size : (i + 1) * hidenchunkSize;
				// workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossSub, this, start, end, klbstart, klbend, std::ref(lossesTmp[i])));
			}
			for (int k = 0; k < iterator; k++)
				;// workers[k]->wait();

			float result = 0.0;
			for (size_t i = 0; i < (ThreadCount > Layers[LayersSize - 1].Size ? Layers[LayersSize - 1].Size : ThreadCount); i++)
				result += lossesTmp[i];

			float regularizerCost = 0.0;
			if (Type == static_cast<int>(NeuralEnums::NetworkType::AutoEncoder) && AutoEncoderType == static_cast<int>(NeuralEnums::AutoEncoderType::Sparce))
			{
				for (size_t w = 1; w < LayersSize; w++)
				{
					for (size_t y = 0; y < (Layers[w].UsingBias ? (Layers[w].Size - 1) * Layers[w - 1].Size : Layers[w].Size * Layers[w - 1].Size); y++)
					{
						regularizerCost += Layers[w].Weights[y] * Layers[w].Weights[y];
					}
				}
			}
			return result + regularizerCost * lambda / 2.0;
		}
	}

	__device__ void CalculateLossBatchSub(int start, int end, float& loss)
	{
		float result = 0.0;
		for (size_t i = start; i < end; i++)
		{
			result += CalculateLossFunctionCuda(LossFunctionType, Layers[LayersSize - 1].GetOutputsBatch(i), Layers[LayersSize - 1].GetTargetsBatch(i), 0, Layers[LayersSize - 1].Size, Layers[LayersSize - 1].Size);
		}
		loss = result;
	}

	__device__ float GetLearningRateMultipliedByGrad(float& gradient, int& iterator, int& j)
	{
		//if (nn.LearningRateType == NeuralEnums::LearningRateType::Static)
		//	return nn.LearningRate * gradient;
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaGrad)
		//	return AdaGrad(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaDelta)
		//	return AdaDelta(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].Parameters, gradient, j) * gradient;
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::Adam)
		//	return Adam(nn, gradient, j, iterator);
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdaMax)
		//	return AdaMax(nn, gradient, j, iterator);
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::AdamMod)
		//	return AdamMod(nn, gradient, j, iterator);
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::RMSProp)
		//	return RMSProp(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
		//else if (nn.LearningRateType == NeuralEnums::LearningRateType::GuraMethod)
		//	return GuraMethod(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].LearningRates, gradient, j, nn.LearningRate) * gradient;
		switch (LearningRateType)
		{
		case static_cast<int>(NeuralEnums::LearningRateType::Static):
			return LearningRate * gradient;
		case static_cast<int>(NeuralEnums::LearningRateType::AdaGrad):
			return AdaGrad(Layers[iterator].GradientsLR, gradient, j) * gradient;
			//case NeuralEnums::LearningRateType::AdaDelta: 
			//	return AdaDelta(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].Parameters, gradient, j) * gradient;
			//	//following 3 methods does not require gradient multiplication
			//case NeuralEnums::LearningRateType::Adam: 
			//	return Adam(nn, gradient, j, iterator);
			//case NeuralEnums::LearningRateType::AdaMax: 
			//	return AdaMax(nn, gradient, j, iterator);
			//case NeuralEnums::LearningRateType::AdamMod: 
			//	return AdamMod(nn, gradient, j, iterator);
			//case NeuralEnums::LearningRateType::RMSProp: 
			//	return RMSProp(nn.Layers[iterator].GradientsLR, gradient, j) * gradient;
			//case NeuralEnums::LearningRateType::GuraMethod: 
			//	return GuraMethod(nn.Layers[iterator].GradientsLR, nn.Layers[iterator].LearningRates, gradient, j, nn.LearningRate) * gradient;
		default:
		{
			// throw std::runtime_error("learning rate function not defined");
		}
		}
	}


	__device__ float Adam(float& gradient, int& j, int& iterator)
	{
		float result, param;

		//mt
		Layers[iterator].Parameters[j] = beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
		//vt
		Layers[iterator].GradientsLR[j] = beta2 * Layers[iterator].GradientsLR[j] + (1 - beta2) * gradient * gradient;


		return (LearningRate * Layers[iterator].Parameters[j]) / ((1 - beta1Pow) * (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2Pow)) + epsilon));
	}

	__device__ float AdaGrad(float* gradients, float& gradient, int& j)
	{
		gradients[j] += gradient * gradient;
		return 0.01 / sqrt(gradients[j] + epsilon);
	}


	__device__ float AdaDelta(float* gradients, float* parameters, float& Gradient, int& j)
	{
		float result, param;
		gradients[j] = momentum * gradients[j] + (1 - momentum) * Gradient * Gradient;
		result = sqrt(parameters[j] + epsilon) / sqrt(gradients[j] + epsilon);
		param = result * Gradient;
		parameters[j] = momentum * parameters[j] + (1 - momentum) * param * param;
		return result;
	}

	__device__ float AdamMod(float& Gradient, int& j, int& iterator)
	{
		float result, param;
		float prelim = (1 - momentum) * Gradient;

		Layers[iterator].GradientsLR[j] = momentum * Layers[iterator].GradientsLR[j] + prelim * Gradient;
		Layers[iterator].Parameters[j] = momentum * Layers[iterator].Parameters[j] + prelim;

		return (LearningRate * Layers[iterator].Parameters[j] / (1 - beta1)) / (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2)) + epsilon);
	}


	__device__ float AdaMax(float& gradient, int& j, int& iterator)
	{
		float result, param;

		//mt
		Layers[iterator].Parameters[j] = beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
		//vt
		Layers[iterator].GradientsLR[j] = (beta2 * Layers[iterator].GradientsLR[j] > abs(gradient) ? beta2 * Layers[iterator].GradientsLR[j] : abs(gradient));


		return (LearningRate * Layers[iterator].Parameters[j]) / ((1 - beta1Pow) * Layers[iterator].GradientsLR[j]);
	}

	__device__ float RMSProp(float* gradients, float& gradient, int& j)
	{
		gradients[j] = momentum * gradients[j] + (1 - momentum) * gradient * gradient;
		return startingLearningRate / sqrt(gradients[j] + epsilon);
	}
	__device__ float DifferentiateLossWithCuda(float& output, float& target, int& function, int size)
	{
		switch (function)
		{
		case 0: return output - target; // NeuralEnums::LossFunctionType::MeanSquaredError
		//case 10: return BinaryCrossentropyDerivative(output, target, size); //NeuralEnums::LossFunctionType::BinaryCrossentropy
		//case 11: return KullbackLeiblerDivergenceDerivative(output, target); //NeuralEnums::LossFunctionType::KullbackLeiblerDivergence
		default:
			break;
		}
	}

};

#endif //NEURALNETWORKCUDA_H

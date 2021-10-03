#ifndef NEURALNETWORKCUDA_H
#define NEURALNETWORKCUDA_H
#include <math.h>
#include <string>
#include <condition_variable>
#include "enums.h"
#include "LayerCuda.cuh"
#include "ActivationFunctionsCuda.cuh"
#include "LossFunctionsCuda.cuh"
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

	//weight decay parameter for sparce autoencoder
	float lambda = 0.7;
	__host__ __device__ NeuralNetworkCuda() {}


	__device__ void PropagateBackThreaded()
	{//
		//ClearNetwork();
		// PopagateBackDelegateBatch2(0, 1, vector);
		if (LearningRateType == static_cast<int>(NeuralEnums::LearningRateType::Adam))
		{
			iterations++;
			beta1Pow = pow(0.9, iterations);
			beta2Pow = pow(0.999, iterations);
		}
		//TODO ეს აქ არ უნდა იყოს
		if (BatchSize > 1)
		{
			int chunkSize = BatchSize / ThreadCount;
			int idx = 0;
			//for (int i = LayersSize - 1; i >= 1; i--)
			//{
			//	if (Layers[i].UsingBias && Layers[i].DropOutSize > 0)
			//	{
			//		delete[](Layers[i].IndexVector);
			//		Layers[i].IndexVectorSize = 0;
			//	}
			//}
			for (int i = 0; i < ThreadCount; i++)
			{
				idx = chunkSize * i;

				//workers[i]->doAsync(std::bind(&NeuralNetwork::PropagateBackDelegateBatch, this, idx, idx + chunkSize, i));
			}
			for (int k = 0; k < ThreadCount; k++)
				;//workers[k]->wait();
			CalculateWeightsBatch();
		}
		else
		{
			for (unsigned int i = LayersSize - 1; i >= 1; i--)
			{

				int Size = Layers[i].Size;
				int chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
				int iterator = ThreadCount > Size ? Size : ThreadCount;
				float** tmpOutputs;
				tmpOutputs = new float* [iterator];

				for (int l = 0; l < iterator; l++)
				{
					int start = l * chunkSize;
					int end = (l + 1) == iterator ? Size : (l + 1) * chunkSize;
					tmpOutputs[l] = new float[Layers[i - 1].Size];
					for (int q = 0; q < Layers[i - 1].Size; q++)
					{
						tmpOutputs[l][q] = 0;
					}
					// workers[l]->doAsync(std::bind(&NeuralNetwork::PropagateBackDelegate, this, i, start, end, tmpOutputs[l]));
				}

				for (int k = 0; k < iterator; k++)
					;//workers[k]->wait();
				if (i != 1)
				{
					for (int f = 0; f < Layers[i - 1].Size; f++)
						Layers[i - 1].Outputs[f] = 0;

					for (int g = 0; g < iterator; g++)
						for (int f = 0; f < Layers[i - 1].Size; f++)
						{
							Layers[i - 1].Outputs[f] += tmpOutputs[g][f];
							tmpOutputs[g][f] = 0;
						}
				}
				for (int v = 0; v < iterator; v++)
				{
					delete[](tmpOutputs[v]);
				}
				delete[](tmpOutputs);
			}
		}
	}

	__device__ void PropagateBackDelegateBatch(int start, int end, int threadNum)
	{
		int numberIndex = 0;
		float* outputsTemp;
		float* inputs;
		int pLS = 0;
		int biasShift = 0;
		float gradient;
		float gradientTemp;

		for (int batch = start; batch < end; batch++)
		{
			for (int i = LayersSize - 1; i >= 1; i--)
			{

				pLS = Layers[i - 1].Size;
				biasShift = Layers[i].UsingBias ? 1 : 0;
				outputsTemp = new float[Layers[i - 1].Size];
				for (int v = 0; v < Layers[i - 1].Size; v++)
				{
					outputsTemp[v] = 0;
				}
				int j, p;
				for (int jj = 0; jj < Layers[i].IndexVectorSize; jj++)
				{
					j = Layers[i].IndexVector[jj];
					// Output ლეიერი
					if (i == LayersSize - 1)
						Layers[i].GetOutputsBatch(batch, j) = DifferentiateLossWithCuda(Layers[i].GetOutputsBatch(batch, j), Layers[i].GetTargetsBatch(batch, j), LossFunctionType, Layers[i].Size);
					Layers[i].GetInputsBatch(batch, j) = Layers[i].GetOutputsBatch(batch, j) * DifferentiateWithCuda(Layers[i].GetInputsBatch(batch, j), Layers[i].ActivationFunction, Layers[i].GetInputsBatch(batch), Layers[i].DropoutNeurons);

					for (int pp = 0; pp < Layers[i - 1].IndexVectorSize; pp++)
					{
						p = Layers[i - 1].IndexVector[pp];
						numberIndex = pLS * (j - biasShift) + p;
						if (i != 1)
							outputsTemp[p] += Layers[i].GetInputsBatch(batch, j) * Layers[i].Weights[numberIndex];
						Layers[i].GetGradientsBatch(threadNum, numberIndex) += Layers[i].GetInputsBatch(batch, j) * Layers[i - 1].GetOutputsBatch(batch, p);// gradient;
					}
					//
				}
				if (i != 1) //ამის ოპტიმიზაცია შეიძლება 
					for (int p = Layers[i - 1].UsingBias ? 1 : 0; p < pLS; p++)
					{
						if (Layers[i - 1].DropoutNeurons[p])
							continue;
						Layers[i - 1].GetOutputsBatch(batch, p) = outputsTemp[p];
					}
				gradient = 0;
				gradientTemp = 0;
				delete[](outputsTemp);
			}
		}
	}


	__device__ void  CalculateWeightsBatch()
	{
		for (unsigned int i = LayersSize - 1; i >= 1; i--)
		{

			int Size = Layers[i].IndexVectorSize;
			int chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
			int iterator = ThreadCount > Size ? Size : ThreadCount;

			for (int l = 0; l < iterator; l++)
			{

				int start = l * chunkSize;
				int end = (l + 1) == iterator ? Size : (l + 1) * chunkSize;
				start = (start == 0 && Layers[i].UsingBias ? 1 : start);
				/*workers[l]->doAsync(std::bind(&NeuralNetwork::CalculateWeightsBatchSub, this, i, Layers[i - 1].IndexVector, Layers[i - 1].IndexVectorSize,
					start, end, Layers[i - 1].UsingBias));*/
			}
			for (int k = 0; k < iterator; k++)
				;// workers[k]->wait();
		}
	}
	__device__ void  CalculateWeightsBatchSub(int i, int* prevLayerIndex, int prevLayerIndexSize, int start, int end, bool prevLayerUsingBias)
	{
		float gradient = 0;
		int numberIndex = 0;
		int pLS = Layers[i - 1].Size;
		int biasShift = Layers[i].UsingBias ? 1 : 0;
		// todo მაინც გასარკვევია
		//if (prevLayerUsingBias)
		//	prevLayerIndex.insert(prevLayerIndex.begin(), 0);
		int p;
		for (int j = start; j < end; j++)
		{
			for (int pp = 0; pp < prevLayerIndexSize; pp++)
			{
				p = prevLayerIndex[pp];
				numberIndex = pLS * (j - biasShift) + p;
				for (int t = 0; t < ThreadCount; t++)
				{
					gradient += Layers[i].GetGradientsBatch(t, numberIndex);
					Layers[i].GetGradientsBatch(t, numberIndex) = 0;
				}
				gradient /= BatchSize;
				Layers[i].Weights[numberIndex] -= GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
				gradient = 0;
			}
		}
		gradient = 0;
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

	__device__ void CalculateLossSub(int start, int end, int klbstart, int  klbend, float& loss)
	{
		float result = 0.0;
		float klbResult = 0.0;
		for (size_t i = start; i < end; i++)
		{
			result += CalculateLossFunctionCuda(LossFunctionType, Layers[LayersSize - 1].Outputs, Layers[LayersSize - 1].Targets, start, end, Layers[LayersSize - 1].Size);
		}

		if (Type == static_cast<int>(NeuralEnums::NetworkType::AutoEncoder) && AutoEncoderType == static_cast<int>(NeuralEnums::AutoEncoderType::Sparce))
		{
			klbResult += KullbackLeiblerDivergenceCuda(Layers[1].RoHat, ro, klbstart, klbend);
		}
		loss = result + klbResult;
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

};

#endif //NEURALNETWORKCUDA_H

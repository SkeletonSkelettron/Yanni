#include "neuralNetwork.h"
#include "statisticFunctions.h"
#include <math.h>
#include <algorithm>

size_t pLS_ = 0;
size_t biasShift_ = 0;
size_t curLayerSize = 0;
void NeuralNetwork::NeuralNetworkInit()
{
	iterations = 0;
	beta1Pow = 0.9f;
	beta2Pow = 0.999f;
	betaAELR = 0.001f;
	ro = -0.9f;
	for (size_t i = 0; i < ThreadCount; i++)
	{
		workers.push_back(new WorkerThread());
	}
	GradientsTemp = new float** [ThreadCount];
	lossesTmp = new float[ThreadCount > Layers[LayersSize - 1].Size ? Layers[LayersSize - 1].Size : ThreadCount];
	lambda = 0.7f;
}

void NeuralNetwork::ShuffleDropoutsPlain()
{
	for (int k = 1; k < LayersSize; k++)
	{
		int biasShift = Layers[k].UsingBias ? 1 : 0;
		if (Layers[k].DropOutSize > 0 && Layers[k].LayerType == NeuralEnums::LayerType::HiddenLayer)
		{
			srand((unsigned int)time(NULL));
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

void NeuralNetwork::InitializeWeights()
{
	srand((unsigned int)time(NULL));
	Layers[0].WeightsSize = 1;
	Layers[0].Weights = new float[1]{ 0 };
	for (int i = 1; i < LayersSize; i++)
	{
		size_t size = Layers[i - 1].Size * (Layers[i].Size - (Layers[i].UsingBias ? 1.0f : 0.0f));
		size_t StartIndex = Layers[i].UsingBias ? 0 : 1;
		Layers[i].Weights = new float[size] {0};
		Layers[i].TempWeights = new float[size] {0};
		if (BatchSize > 1)
		{
			Layers[i].GradientsBatch = new float* [ThreadCount];
			for (int v = 0; v < ThreadCount; v++)
			{
				Layers[i].GradientsBatch[v] = new float[size] {0}; for (size_t b = 0; b < size; b++)Layers[i].GradientsBatch[v][b] = 0;
			}
		}
		else
			Layers[i].Gradients = new float[size] {0};
		Layers[i].GradientsLR = new float[size] {0};
		Layers[i].Parameters = new float[size] {0};
		Layers[i].WeightsSize = size;
		Layers[i].MultipliedSums = new float[size] {0};

		for (int j = 0; j < size; j++)
		{
			Layers[i].Weights[j] = Layers[i].UsingBias && j % Layers[i - 1].Size == 0
				? 1.0
				: rand() % 100;
		}
		if (Layers[i].ActivationFunction == NeuralEnums::ActivationFunction::Sigmoid)
		{
			int tmp[2];
			float start = -1.0;
			float end = 1.0;
			StandartizeLinearContract(Layers[i].Weights, size, tmp, start, end);
		}
		else
		{
			int tmp[2];
			float start = -0.07;
			float end = 0.07;
			StandartizeLinearContract(Layers[i].Weights, size, tmp, start, end);
		}
		if (Layers[i - 1].UsingBias)
			for (int j = 0; j < size; j++)
			{
				if (j % Layers[i - 1].Size == 0)
					Layers[i].Weights[j] = 1L;
			}
		//std::ofstream oData;
		//oData.open("weights" + std::to_string(i) + ".txt");
		//for (int count = 0; count < size; count++) {
		//	oData << std::setprecision(100) << Layers[i].Weights[count] << std::endl;
		//}

		/*std::ifstream inData;
		inData.open("weights" + std::to_string(i) + ".txt");
		for (size_t count = 0; count < Layers[i - 1].Size * (Layers[i].Size - (Layers[i].UsingBias ? 1 : 0)); count++) {
			inData >> std::setprecision(100) >> Layers[i].Weights[count];
		}*/
	}
}

float NeuralNetwork::PropagateForwardThreaded(bool training, bool countingRohat)
{
	//if (training)
		//ShuffleDropoutsPlain();
	for (size_t k = 1; k < LayersSize - (countingRohat ? 1 : 0); k++)
	{
		Layers[k].CalculateInputsThreaded(Layers[k - 1].Outputs, Layers[k - 1].Size, Layers[k - 1].OutputsBatch, Layers[k - 1].IndexVectorForNextLayer, Layers[k - 1].IndexVectorForNextLayerSize, training, ThreadCount, workers);
		Layers[k].CalculateOutputsThreaded(ThreadCount, training, countingRohat, workers);
	}
	if (this->LossCalculation == NeuralEnums::LossCalculation::Full && !countingRohat)
		return CalculateLoss(training);
	return -1;
}


void NeuralNetwork::PropagateBackThreaded()
{//https://hmkcode.github.io/ai/backpropagation-step-by-step/
	//ClearNetwork();
	// PopagateBackDelegateBatch2(0, 1, vector);
	if (LearningRateType == NeuralEnums::LearningRateType::Adam)
	{
		iterations++;
		beta1Pow = (float)pow(0.9f, iterations);
		beta2Pow = (float)pow(0.999f, iterations);
	}
	//TODO ეს აქ არ უნდა იყოს
	if (BatchSize > 1)
	{
		size_t chunkSize = BatchSize / ThreadCount;
		size_t idx = 0;

		for (size_t i = 0; i < ThreadCount; i++)
		{
			idx = chunkSize * i;

			workers[i]->doAsync(std::bind(&NeuralNetwork::PropagateBackDelegateBatch, this, idx, idx + chunkSize, i));
		}
		for (int k = 0; k < ThreadCount; k++)
			workers[k]->wait();
		CalculateWeightsBatch();
	}
	else
	{
		for (size_t i = LayersSize - 1; i >= 1; i--)
		{
			pLS_ = Layers[i - 1].Size;
			biasShift_ = Layers[i].UsingBias ? 1 : 0;
			curLayerSize = Layers[i].Size;

			size_t Size = Layers[i].IndexVectorSize;
			size_t chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
			size_t threadsNum = ThreadCount > Size ? Size : ThreadCount;

			for (size_t threadId = 0; threadId < threadsNum; threadId++)
			{
				size_t start = threadId * chunkSize;
				size_t end = (threadId + 1) == threadsNum ? Size : (threadId + 1) * chunkSize;
				// PropagateBackDelegate ზე გადართვისას int Size = Layers[i].IndexVectorSize; უნდა მივუთითო
				workers[threadId]->doAsync(std::bind(&NeuralNetwork::PropagateBackDelegate, this, i, start, end));
			}

			for (size_t k = 0; k < threadsNum; k++)
				workers[k]->wait();
		}
	}
}

//66 წამი
void NeuralNetwork::PropagateBackDelegate(int i, size_t start, size_t end)
{
	size_t numberIndex = 0;
	size_t nls = Layers[i + 1].Size;
	size_t j = 0, p = 0, l = 0;
	for (size_t jj = start; jj < end; jj++)
	{
		j = Layers[i].IndexVector[jj];
		// Output ლეიერი
		if (i == LayersSize - 1)
			Layers[i].Outputs[j] = DifferentiateLossWith(Layers[i].Outputs[j], Layers[i].Target[j], LossFunctionType, Layers[i].Size);
		else
		{
			size_t nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;
			Layers[i].Outputs[j] = 0;

			for (size_t ll = Layers[i + 1].IndexVectorSize; ll--;)
			{
				l = Layers[i + 1].IndexVector[ll];
				Layers[i].Outputs[j] += Layers[i + 1].Inputs[l] * Layers[i + 1].TempWeights[(l - nextLayerBiasShift) * curLayerSize + j];
			}
		}
		Layers[i].Inputs[j] = Layers[i].Outputs[j] * DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction, Layers[i].Inputs, Layers[i].DropoutNeurons);

		for (size_t pp = Layers[i - 1].IndexVectorForNextLayerSize; pp--;)
		{
			p = Layers[i - 1].IndexVectorForNextLayer[pp];
			numberIndex = pLS_ * (j - biasShift_) + p;
			if (i != 1)
				Layers[i].TempWeights[numberIndex] = Layers[i].Weights[numberIndex];
			//Layers[i].Gradients[numberIndex] = ... if gradient optimization is needed
			Layers[i].Weights[numberIndex] -= Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] * LearningRate;// Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] ეს არის გრადიენტი GetLearningRateMultipliedByGrad(gradient/*Layers[i].Gradients[numberIndex]*/, i, numberIndex);
		}
	}
}



void NeuralNetwork::PropagateBackDelegateNew(int i, size_t start, size_t end)
{
	size_t numberIndex = 0;
	size_t pLS = Layers[i - 1].Size;
	int biasShift = Layers[i].UsingBias ? 1 : 0;
	float gradient;
	int j = 0, p = 0, n = 0;
	for (size_t jj = start; jj < end; jj++)
	{
		j = Layers[i].IndexVector[jj];
		// Output ლეიერი
		if (i == LayersSize - 1)
		{
			Layers[i].Outputs[j] = DifferentiateLossWith(Layers[i].Outputs[j], Layers[i].Target[j], LossFunctionType, Layers[i].Size);

			Layers[i].Inputs[j] = DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction, Layers[i].Inputs, Layers[i].DropoutNeurons);

			// Output ლეიერში წონების დაკორექტირება
			for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++)
			{
				p = Layers[i - 1].IndexVectorForNextLayer[pp];
				numberIndex = pLS * (j - biasShift) + p;
				Layers[i].MultipliedSums[numberIndex] = Layers[i].Weights[numberIndex] * Layers[i].Inputs[j] * Layers[i].Outputs[j];
				//gradient = Layers[i].Outputs[j] * Layers[i].Inputs[j] * Layers[i - 1].Outputs[p];

				Layers[i].Weights[numberIndex] -= Layers[i].Outputs[j] * Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
			}
		}
		else
		{
			Layers[i].Inputs[j] = DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction, Layers[i].Inputs, Layers[i].DropoutNeurons);

			float sum = 0;
			size_t nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;

			//შემდეგი ლეიერიდან უნდა აიღოს შესაბამისი ჯამები
			for (size_t n = Layers[i + 1].UsingBias ? 1 : 0; n < Layers[i + 1].Size; n++)
			{
				sum += Layers[i + 1].MultipliedSums[(n - nextLayerBiasShift) * Layers[i].Size + j];
			}

			//მიმდინარე ნეირონის შესაბამისი წონების განახლება

			float mult = Layers[i].Inputs[j] * sum;

			if (i != 1)
				for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++)
				{
					// mult * Layers[i - 1].Outputs[n] არის gradient
					n = Layers[i - 1].IndexVectorForNextLayer[pp];
					numberIndex = pLS * (j - biasShift) + n;
					Layers[i].MultipliedSums[numberIndex] = Layers[i].Weights[numberIndex] * mult;
					gradient = mult * Layers[i - 1].Outputs[n];

					Layers[i].Weights[numberIndex] -= mult * Layers[i - 1].Outputs[n] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
				}
			else
				for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++)
				{
					n = Layers[i - 1].IndexVectorForNextLayer[pp];
					numberIndex = pLS * (j - biasShift) + n;
					gradient = mult * Layers[i - 1].Outputs[n];
					Layers[i].Weights[numberIndex] -= mult * Layers[i - 1].Outputs[n] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
				}
		}
	}
}

// ეს რეალიზაცია, დროპაუთების გარეშე, ყველაზე სწრაფია 43 წამი უნდება
//void NeuralNetwork::PropagateBackDelegateNew(int i, int start, int end)
//{
//	int numberIndex = 0;
//	int pLS = Layers[i - 1].Size;
//	int biasShift = Layers[i].UsingBias ? 1 : 0;
//	start = start == 0 && Layers[i].UsingBias ? 1 : start;
//	float gradient;
//	float gradientTemp;
//	for (long int j = start; j < end; j++)
//	{
//		if (Layers[i].DropoutNeurons[j])
//			continue;
//		// Output ლეიერი
//		if (i == LayersSize - 1)
//		{
//			Layers[i].Outputs[j] = DifferentiateLossWith(Layers[i].Outputs[j], Layers[i].Target[j], LossFunctionType, Layers[i].Size);
//
//			Layers[i].Inputs[j] = DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction, Layers[i].Inputs, Layers[i].DropoutNeurons);
//
//			// Output ლეიერში წონების დაკორექტირება
//			for (long int p = 0; p < pLS; p++)
//			{
//				if (Layers[i - 1].DropoutNeurons[p])
//					continue;
//				numberIndex = pLS * (j - biasShift) + p;
//				Layers[i].MultipliedSums[numberIndex] = Layers[i].Weights[numberIndex] * Layers[i].Inputs[j] * Layers[i].Outputs[j];
//				//gradient = Layers[i].Outputs[j] * Layers[i].Inputs[j] * Layers[i - 1].Outputs[p];
//
//				Layers[i].Weights[numberIndex] -= Layers[i].Outputs[j] * Layers[i].Inputs[j] * Layers[i - 1].Outputs[p] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
//			}
//		}
//		else
//		{
//			Layers[i].Inputs[j] = DifferentiateWith(Layers[i].Inputs[j], Layers[i].ActivationFunction, Layers[i].Inputs, Layers[i].DropoutNeurons);
//
//			float sum = 0;
//			int nextLayerBiasShift = Layers[i + 1].UsingBias ? 1 : 0;
//
//			//შემდეგი ლეიერიდან უნდა აიღოს შესაბამისი ჯამები
//			for (long int n = Layers[i + 1].UsingBias ? 1 : 0; n < Layers[i + 1].Size; n++)
//			{
//				if (Layers[i + 1].DropoutNeurons[n])
//					continue;
//				sum += Layers[i + 1].MultipliedSums[(n - nextLayerBiasShift) * Layers[i].Size + j];
//			}
//
//			//მიმდინარე ნეირონის შესაბამისი წონების განახლება
//
//			float mult = Layers[i].Inputs[j] * sum;
//
//			if (i != 1)
//				for (long int n = 0; n < pLS; n++)
//				{
//					// mult * Layers[i - 1].Outputs[n] არის gradient
//					if (Layers[i - 1].DropoutNeurons[n])
//						continue;
//					numberIndex = pLS * (j - biasShift) + n;
//					Layers[i].MultipliedSums[numberIndex] = Layers[i].Weights[numberIndex] * mult;
//					gradient = mult * Layers[i - 1].Outputs[n];
//
//					Layers[i].Weights[numberIndex] -= mult * Layers[i - 1].Outputs[n] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
//				}
//			else
//				for (long int n = 0; n < pLS; n++)
//				{
//					if (Layers[i - 1].DropoutNeurons[n])
//						continue;
//					numberIndex = pLS * (j - biasShift) + n;
//					gradient = mult * Layers[i - 1].Outputs[n];
//					Layers[i].Weights[numberIndex] -= mult * Layers[i - 1].Outputs[n] * LearningRate;// GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
//				}
//		}
//	}
//}

void NeuralNetwork::PropagateBackDelegateBatch(size_t start, size_t end, int threadNum)
{
	size_t numberIndex = 0;
	float* outputsTemp;
	size_t pLS = 0;
	size_t biasShift = 0;
	float gradient;
	float gradientTemp;

	for (size_t batch = start; batch < end; batch++)
	{
		for (size_t i = LayersSize - 1; i >= 1; i--)
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
					Layers[i].OutputsBatch[batch][j] = DifferentiateLossWith(Layers[i].OutputsBatch[batch][j], Layers[i].TargetsBatch[batch][j], LossFunctionType, Layers[i].Size);
				Layers[i].InputsBatch[batch][j] = Layers[i].OutputsBatch[batch][j] * DifferentiateWith(Layers[i].InputsBatch[batch][j], Layers[i].ActivationFunction, Layers[i].InputsBatch[batch], Layers[i].DropoutNeurons);

				for (int pp = 0; pp < Layers[i - 1].IndexVectorForNextLayerSize; pp++)
				{
					p = Layers[i - 1].IndexVectorForNextLayer[pp];
					numberIndex = pLS * (j - biasShift) + p;
					if (i != 1)
						outputsTemp[p] += Layers[i].InputsBatch[batch][j] * Layers[i].Weights[numberIndex];
					Layers[i].GradientsBatch[threadNum][numberIndex] += Layers[i].InputsBatch[batch][j] * Layers[i - 1].OutputsBatch[batch][p];// gradient;
				}
				//
			}
			if (i != 1) //ამის ოპტიმიზაცია შეიძლება 
				for (int p = /*Layers[i - 1].UsingBias ? 1 :*/ 0; p < pLS; p++)
				{
					if (Layers[i - 1].DropoutNeurons[p])
						continue;
					Layers[i - 1].OutputsBatch[batch][p] = outputsTemp[p];
				}
			gradient = 0;
			gradientTemp = 0;
			delete[](outputsTemp);
		}
	}
}


void  NeuralNetwork::CalculateWeightsBatch()
{
	for (size_t i = LayersSize - 1; i >= 1; i--)
	{

		int Size = Layers[i].IndexVectorSize;
		int chunkSize = Size / ThreadCount == 0 ? 1 : Size / ThreadCount;
		int iterator = ThreadCount > Size ? Size : ThreadCount;

		for (int l = 0; l < iterator; l++)
		{

			size_t start = l * chunkSize;
			size_t end = (l + 1) == iterator ? Size : (l + 1) * chunkSize;
			start = (start == 0 && Layers[i].UsingBias ? 1 : start);
			workers[l]->doAsync(std::bind(&NeuralNetwork::CalculateWeightsBatchSub, this, i, Layers[i - 1].IndexVectorForNextLayer, Layers[i - 1].IndexVectorForNextLayerSize,
				start, end, Layers[i - 1].UsingBias));
		}
		for (int k = 0; k < iterator; k++)
			workers[k]->wait();
	}
}
void  NeuralNetwork::CalculateWeightsBatchSub(int i, size_t* prevLayerIndex, size_t prevLayerIndexSize, size_t start, size_t end, bool prevLayerUsingBias)
{
	float gradient = 0;
	size_t numberIndex = 0;
	size_t pLS = Layers[i - 1].Size;
	size_t biasShift = Layers[i].UsingBias ? 1 : 0;
	size_t p;
	for (size_t j = start; j < end; j++)
	{
		for (size_t pp = 0; pp < prevLayerIndexSize; pp++)
		{
			p = prevLayerIndex[pp];
			numberIndex = pLS * (j - biasShift) + p;
			for (size_t t = 0; t < ThreadCount; t++)
			{
				gradient += Layers[i].GradientsBatch[t][numberIndex];
				Layers[i].GradientsBatch[t][numberIndex] = 0;
			}
			gradient /= BatchSize;
			Layers[i].Weights[numberIndex] -= GetLearningRateMultipliedByGrad(gradient, i, numberIndex);
			gradient = 0;
		}
	}
	gradient = 0;
}

void  NeuralNetwork::PrepareForTesting()
{
	for (size_t k = 0; k < LayersSize; k++)
	{
		size_t biasShift = Layers[k].UsingBias ? 1 : 0;
		Layers[k].IndexVectorSize = Layers[k].Size - biasShift;
		Layers[k].IndexVector = new size_t[(int)Layers[k].IndexVectorSize];
		Layers[k].IndexVectorForNextLayer = new size_t[Layers[k].Size];
		Layers[k].IndexVectorForNextLayerSize = Layers[k].Size;
		Layers[k].IndexVectorForNextLayer[0] = (size_t)0;
		for (int i = biasShift; i < Layers[k].Size; i++)
		{
			Layers[k].IndexVector[i - biasShift] = i;
			Layers[k].IndexVectorForNextLayer[i] = i;
		}
	}
}

float NeuralNetwork::CalculateLoss(bool& training)
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
		float result;
		for (int i = 0; i < ThreadCount; i++)
		{
			idx = chunkSize * i;

			workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossBatchSub, this, idx, idx + chunkSize, std::ref(losses[i])));
		}
		for (int k = 0; k < ThreadCount; k++)
			workers[k]->wait();
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
		size_t hidenchunkSize = 0;
		if (Type == NeuralEnums::NetworkType::AutoEncoder && AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce)
		{
			size_t hidenchunkSize = Layers[LayersSize - 2].Size / ThreadCount == 0 ? 1 : Layers[LayersSize - 2].Size / ThreadCount;
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
			workers[i]->doAsync(std::bind(&NeuralNetwork::CalculateLossSub, this, start, end, klbstart, klbend, std::ref(lossesTmp[i])));
		}
		for (int k = 0; k < iterator; k++)
			workers[k]->wait();

		float result = 0.0;
		for (size_t i = 0; i < (ThreadCount > Layers[LayersSize - 1].Size ? Layers[LayersSize - 1].Size : ThreadCount); i++)
			result += lossesTmp[i];

		float regularizerCost = 0.0;
		if (Type == NeuralEnums::NetworkType::AutoEncoder && AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce)
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

void NeuralNetwork::CalculateLossBatchSub(size_t start, size_t end, float& loss)
{
	float result = 0.0;
	for (size_t i = start; i < end; i++)
	{
		result += CalculateLossFunction(LossFunctionType, Layers[LayersSize - 1].OutputsBatch[i], Layers[LayersSize - 1].TargetsBatch[i], 0, Layers[LayersSize - 1].Size, Layers[LayersSize - 1].Size);
	}
	loss = result;
}

void NeuralNetwork::CalculateLossSub(size_t start, size_t end, size_t klbstart, size_t  klbend, float& loss)
{
	float result = 0.0;
	float klbResult = 0.0;
	for (size_t i = start; i < end; i++)
	{
		result += CalculateLossFunction(LossFunctionType, Layers[LayersSize - 1].Outputs, Layers[LayersSize - 1].Target, start, end, Layers[LayersSize - 1].Size);
	}

	if (Type == NeuralEnums::NetworkType::AutoEncoder && AutoEncoderType == NeuralEnums::AutoEncoderType::Sparce)
	{
		klbResult += KullbackLeiblerDivergence(Layers[1].RoHat, ro, klbstart, klbend);
	}
	loss = result + klbResult;
}



float NeuralNetwork::GetLearningRateMultipliedByGrad(float& gradient, int& iterator, size_t& j)
{
	switch (LearningRateType)
	{
	case NeuralEnums::LearningRateType::Static:
	{
		return LearningRate * gradient;
		break;
	}
	case NeuralEnums::LearningRateType::AdaGrad:
	{
		return AdaGrad(Layers[iterator].GradientsLR, gradient, j) * gradient;
		break;
	}
	case NeuralEnums::LearningRateType::AdaDelta:
	{
		return AdaDelta(Layers[iterator].GradientsLR, Layers[iterator].Parameters, gradient, j) * gradient;
		break;
	}
	//following 3 methods does not require gradient multiplication
	case NeuralEnums::LearningRateType::Adam:
	{
		return Adam(gradient, j, iterator);
		break;
	}
	case NeuralEnums::LearningRateType::AdaMax:
	{
		return AdaMax(gradient, j, iterator);
		break;
	}
	case NeuralEnums::LearningRateType::AdamMod:
	{
		return AdamMod(gradient, j, iterator);
		break;
	}
	case NeuralEnums::LearningRateType::RMSProp:
	{
		return RMSProp(Layers[iterator].GradientsLR, gradient, j) * gradient;
		break;
	}
	default:
	{
		throw std::runtime_error("learning rate function not defined");
		break;
	}
	}
}


float NeuralNetwork::Adam(float& gradient, size_t& j, int& iterator)
{
	float result, param;

	//mt
	Layers[iterator].Parameters[j] = beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
	//vt
	Layers[iterator].GradientsLR[j] = beta2 * Layers[iterator].GradientsLR[j] + (1 - beta2) * gradient * gradient;


	return (LearningRate * Layers[iterator].Parameters[j]) / ((1 - beta1Pow) * (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2Pow)) + epsilon));
}

float NeuralNetwork::AdaGrad(float* gradients, float& gradient, size_t& j)
{
	gradients[j] += gradient * gradient;
	return 0.01 / sqrt(gradients[j] + epsilon);
}


float NeuralNetwork::AdaDelta(float* gradients, float* parameters, float& Gradient, size_t& j)
{
	float result, param;
	gradients[j] = momentum * gradients[j] + (1 - momentum) * Gradient * Gradient;
	result = sqrt(parameters[j] + epsilon) / sqrt(gradients[j] + epsilon);
	param = result * Gradient;
	parameters[j] = momentum * parameters[j] + (1 - momentum) * param * param;
	return result;
}

float NeuralNetwork::AdamMod(float& Gradient, size_t& j, int& iterator)
{
	float result, param;
	float prelim = (1 - momentum) * Gradient;

	Layers[iterator].GradientsLR[j] = momentum * Layers[iterator].GradientsLR[j] + prelim * Gradient;
	Layers[iterator].Parameters[j] = momentum * Layers[iterator].Parameters[j] + prelim;

	return (LearningRate * Layers[iterator].Parameters[j] / (1 - beta1)) / (sqrt(Layers[iterator].GradientsLR[j] / (1 - beta2)) + epsilon);
}


float NeuralNetwork::AdaMax(float& gradient, size_t& j, int& iterator)
{
	float result, param;

	//mt
	Layers[iterator].Parameters[j] = beta1 * Layers[iterator].Parameters[j] + (1 - beta1) * gradient;
	//vt
	Layers[iterator].GradientsLR[j] = std::max(beta2 * Layers[iterator].GradientsLR[j], abs(gradient));


	return (LearningRate * Layers[iterator].Parameters[j]) / ((1 - beta1Pow) * Layers[iterator].GradientsLR[j]);
}

float NeuralNetwork::RMSProp(float* gradients, float& gradient, size_t& j)
{
	gradients[j] = momentum * gradients[j] + (1 - momentum) * gradient * gradient;
	return startingLearningRate / sqrt(gradients[j] + epsilon);
}

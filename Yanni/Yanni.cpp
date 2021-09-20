#include "Yanni.h"


using namespace std;

void test()
{
	float* ar = new float[4];
	//bias test case
	NeuralNetwork neuralNetwork;
	ar[0] = 1.0f;
	ar[1] = 0.9f;
	ar[2] = 0.4f;
	ar[3] = 0.7f;
	std::vector<float> labels;
	float* targetArray;
	std::vector<double> losses;
	targetArray = new float[2];
	targetArray[0] = 1.0f;
	targetArray[1] = 0.01f;
	Layer layerInput(4, NeuralEnums::LayerType::InputLayer, NeuralEnums::ActivationFunction::None, 1, 0);
	Layer layerHidden1(4, NeuralEnums::LayerType::HiddenLayer, NeuralEnums::ActivationFunction::Sigmoid, 1, 0);
	Layer layerHidden2(3, NeuralEnums::LayerType::HiddenLayer, NeuralEnums::ActivationFunction::Sigmoid, 1, 0);
	Layer layerOutput(2, NeuralEnums::LayerType::OutputLayer, NeuralEnums::ActivationFunction::Sigmoid, 0, 0);
	Layer* vecs;
	vecs = new Layer[4];
	layerInput.WeightsSize = 1;
	layerInput.Weights = new float[1];
	vecs[0] = layerInput;
	vecs[1] = layerHidden1;
	vecs[2] = layerHidden2;
	vecs[3] = layerOutput;
	neuralNetwork.Layers = vecs;
	neuralNetwork.LearningRate = 0.6f;
	neuralNetwork.ThreadCount = 1;
	neuralNetwork.Cuda = false;
	neuralNetwork.BatchSize = 1;
	neuralNetwork.LogLoss = NeuralEnums::LogLoss::Full;
	neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::Static;
	neuralNetwork.BalanceType = NeuralEnums::BalanceType::GaussianStandartization;
	neuralNetwork.LossFunctionType = NeuralEnums::LossFunctionType::MeanSquaredError;
	neuralNetwork.LossCalculation = NeuralEnums::LossCalculation::Full;
	neuralNetwork.LayersSize = 4;
	//neuralNetwork.Layers[1].GradientsBatch.resize(1);
	//neuralNetwork.Layers[1].GradientsBatch[0].resize(12);
	neuralNetwork.Layers[1].Gradients = new float[12];
	neuralNetwork.Layers[1].Weights = new float[12];
	neuralNetwork.Layers[1].TempWeights = new float[12];
	neuralNetwork.Layers[1].MultipliedSums = new float[12];
	neuralNetwork.Layers[1].Weights[0] = 1.0f;
	neuralNetwork.Layers[1].Weights[1] = 0.6f;
	neuralNetwork.Layers[1].Weights[2] = 0.7f;
	neuralNetwork.Layers[1].Weights[3] = -0.4f;
	neuralNetwork.Layers[1].Weights[4] = 1.0f;
	neuralNetwork.Layers[1].Weights[5] = -0.8f;
	neuralNetwork.Layers[1].Weights[6] = 0.4f;
	neuralNetwork.Layers[1].Weights[7] = 0.1f;
	neuralNetwork.Layers[1].Weights[8] = 1.0f;
	neuralNetwork.Layers[1].Weights[9] = 0.23f;
	neuralNetwork.Layers[1].Weights[10] = 0.17f;
	neuralNetwork.Layers[1].Weights[11] = 0.16f;

	// neuralNetwork.Layers[2].GradientsBatch.resize(1);
	//neuralNetwork.Layers[2].GradientsBatch[0].resize(8);
	neuralNetwork.Layers[2].Gradients = new float[8];
	neuralNetwork.Layers[2].Weights = new float[8];
	neuralNetwork.Layers[2].TempWeights = new float[8];
	neuralNetwork.Layers[2].MultipliedSums = new float[8];
	neuralNetwork.Layers[2].Weights[0] = 1.0f;
	neuralNetwork.Layers[2].Weights[1] = -0.5f;
	neuralNetwork.Layers[2].Weights[2] = 0.5f;
	neuralNetwork.Layers[2].Weights[3] = -0.2f;
	neuralNetwork.Layers[2].Weights[4] = 1.0f;
	neuralNetwork.Layers[2].Weights[5] = 0.3f;
	neuralNetwork.Layers[2].Weights[6] = -0.46f;
	neuralNetwork.Layers[2].Weights[7] = 0.76f;

	// neuralNetwork.Layers[3].GradientsBatch.resize(1);
	//neuralNetwork.Layers[3].GradientsBatch[0].resize(6);
	neuralNetwork.Layers[3].Gradients = new float[6];
	neuralNetwork.Layers[3].Weights = new float[6];
	neuralNetwork.Layers[3].TempWeights = new float[6];
	neuralNetwork.Layers[3].MultipliedSums = new float[6];
	neuralNetwork.Layers[3].Weights[0] = 1.0f;
	neuralNetwork.Layers[3].Weights[1] = 0.3f;
	neuralNetwork.Layers[3].Weights[2] = 0.4f;
	neuralNetwork.Layers[3].Weights[3] = 1.0f;
	neuralNetwork.Layers[3].Weights[4] = 0.7f;
	neuralNetwork.Layers[3].Weights[5] = 0.92f;
	neuralNetwork.Layers[3].Target = targetArray;

	neuralNetwork.Layers[3].Target = targetArray;
	neuralNetwork.Layers[0].Outputs = ar;
	neuralNetwork.NeuralNetworkInit();

	MnistData* trainingSet;
	trainingSet = new MnistData[1];
	trainingSet[0].set = new float[4];
	trainingSet[0].set[0] = 1.0;
	trainingSet[0].set[1] = (float)0.9;
	trainingSet[0].set[2] = (float)0.4;
	trainingSet[0].set[3] = (float)0.7;
	trainingSet[0].setSize = 4;

	trainingSet[0].label = new float[2];
	trainingSet[0].label[0] = 1.0;
	trainingSet[0].label[1] = (float)0.01;
	trainingSet[0].labelSize = 2;

	neuralNetwork.Layers[1].WeightsSize = 12;
	neuralNetwork.Layers[2].WeightsSize = 8;
	neuralNetwork.Layers[3].WeightsSize = 6;

	//initTrainingAndTestData<double>(neuralNetwork, trainingSet, 1, NULL, 0);
	neuralNetwork.PrepareForTesting();

	// copyNetrworkCuda(neuralNetwork, nullptr, 0, nullptr, 0, false);

	for (size_t i = 0; i < 100; i++)
	{
		auto loss = neuralNetwork.PropagateForwardThreaded(true, false); // პირველი loss უნდა იყოს 0.20739494219121993   float ზე 0.414789855
		neuralNetwork.PropagateBackThreaded();

		//1.0002170802724326
		//0.60019537224518926
		//0.70008683210897293
		//- 0.39984804380929723
		//0.99984718866787936
		//- 0.80013753019890865
		//0.39993887546715173
		//0.099893032067515528
		//0.99955724383635436
		//0.22960151945271889
		//0.16982289753454174
		//0.15969007068544802
		//	1.00021708
		//	0.600195408
		//	0.700086832
		//	- 0.399848044
		//	0.999847174
		//	- 0.800137520
		//	0.399938881
		//	0.0998930335
		//	0.999557257
		//	0.229601517
		//	0.169822901
		//	0.159690067

		losses.push_back(loss);
	}
}
void initNeUnetFromJson(NeuralNetwork& neuralNetwork)
{
	char result[MAX_PATH]{};
	auto rslt = std::string(result, GetModuleFileNameA(NULL, result, MAX_PATH));

	rslt = rslt.substr(0, rslt.find_last_of("\\/"));

	std::ifstream ifs("C:/Users/Misha/source/repos/Yanni/Yanni/netConfig.json");
	if (!ifs.is_open())
		return;

	std::string content((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));
	nlohmann::json json = nlohmann::json::parse(content);

	neuralNetwork.ThreadCount = (size_t)json["ThreadCount"];
	neuralNetwork.LearningRate = json["LearningRate"];

	auto lrType = json["LearningRateType"];
	auto balance = json["Balance"];
	auto lossFunction = json["LossFunction"];
	auto gradient = json["Gradient"];
	auto metrics = json["Metrics"];
	auto losscalc = json["LossCalculation"];
	auto autoEncoderType = json["AutoEncoderType"];
	auto logLoss = json["LogLoss"];
	neuralNetwork.BatchSize = (int)json["BatchSize"];
	neuralNetwork.Cuda = (bool)json["Cuda"];

	auto networkType = json["Type"];

	if (networkType == "Normal")
		neuralNetwork.Type = NeuralEnums::NetworkType::Normal;
	if (networkType == "AutoEncoder")
		neuralNetwork.Type = NeuralEnums::NetworkType::AutoEncoder;

	if (metrics == "None")
		neuralNetwork.Metrics = NeuralEnums::Metrics::None;
	if (metrics == "Test")
		neuralNetwork.Metrics = NeuralEnums::Metrics::TestSet;
	if (metrics == "Full")
		neuralNetwork.Metrics = NeuralEnums::Metrics::Full;

	if (logLoss == "None")
		neuralNetwork.LogLoss = NeuralEnums::LogLoss::None;
	if (logLoss == "Sparce")
		neuralNetwork.LogLoss = NeuralEnums::LogLoss::Sparce;
	if (logLoss == "Full")
		neuralNetwork.LogLoss = NeuralEnums::LogLoss::Full;

	if (autoEncoderType == "Contractive")
		neuralNetwork.AutoEncoderType = NeuralEnums::AutoEncoderType::Contractive;
	if (autoEncoderType == "Denoising")
		neuralNetwork.AutoEncoderType = NeuralEnums::AutoEncoderType::Denoising;
	if (autoEncoderType == "Sparce")
		neuralNetwork.AutoEncoderType = NeuralEnums::AutoEncoderType::Sparce;
	if (autoEncoderType == "UnderComplete")
		neuralNetwork.AutoEncoderType = NeuralEnums::AutoEncoderType::UnderComplete;
	else
		neuralNetwork.AutoEncoderType = NeuralEnums::AutoEncoderType::None;

	if (losscalc == "None")
		neuralNetwork.LossCalculation = NeuralEnums::LossCalculation::None;
	if (losscalc == "Full")
		neuralNetwork.LossCalculation = NeuralEnums::LossCalculation::Full;

	if (lrType == "Static")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::Static;
	if (lrType == "AdaDelta")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::AdaDelta;
	if (lrType == "AdaGrad")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::AdaGrad;
	if (lrType == "Adam")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::Adam;
	if (lrType == "AdamMod")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::AdamMod;
	if (lrType == "cyclic")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::Cyclic;
	if (lrType == "RMSProp")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::RMSProp;
	if (lrType == "GuraMethod")
		neuralNetwork.LearningRateType = NeuralEnums::LearningRateType::GuraMethod;

	if (balance == "GaussianStandartization")
		neuralNetwork.BalanceType = NeuralEnums::BalanceType::GaussianStandartization;
	if (balance == "NormalDistrubution")
		neuralNetwork.BalanceType = NeuralEnums::BalanceType::NormalDistrubution;
	if (balance == "Normalization")
		neuralNetwork.BalanceType = NeuralEnums::BalanceType::Normalization;
	if (balance == "None")
		neuralNetwork.BalanceType = NeuralEnums::BalanceType::None;

	if (lossFunction == "BinaryCrossentropy")
		neuralNetwork.LossFunctionType = NeuralEnums::LossFunctionType::BinaryCrossentropy;
	if (lossFunction == "MeanSquaredError")
		neuralNetwork.LossFunctionType = NeuralEnums::LossFunctionType::MeanSquaredError;
	if (lossFunction == "KullbackLeiblerDivergence")
		neuralNetwork.LossFunctionType = NeuralEnums::LossFunctionType::KullbackLeiblerDivergence;

	if (gradient == "Momentum")
		neuralNetwork.GradientType = NeuralEnums::GradientType::Momentum;
	if (gradient == "Static")
		neuralNetwork.GradientType = NeuralEnums::GradientType::Static;

	neuralNetwork.Layers = new Layer[json["Layers"].size()];
	size_t counter = 0;
	neuralNetwork.LayersSize = json["Layers"].size();
	for (auto& layer : json["Layers"])
	{
		NeuralEnums::LayerType LayerType;
		NeuralEnums::ActivationFunction ActivationFunctionType;
		auto activationFunction = layer["ActivationFunction"];
		auto layerType = layer["Type"];

		float DropuOutSize = layer["DropuOutSize"];
		float bias = layer["Bias"];
		size_t size = layer["Size"];
		if (activationFunction == "MReLU")
			ActivationFunctionType = NeuralEnums::ActivationFunction::MReLU;
		else if (activationFunction == "None")
			ActivationFunctionType = NeuralEnums::ActivationFunction::None;
		else if (activationFunction == "ReLU")
			ActivationFunctionType = NeuralEnums::ActivationFunction::ReLU;
		else if (activationFunction == "Sigmoid")
			ActivationFunctionType = NeuralEnums::ActivationFunction::Sigmoid;
		else if (activationFunction == "SoftMax")
			ActivationFunctionType = NeuralEnums::ActivationFunction::SoftMax;
		else if (activationFunction == "Tanh")
			ActivationFunctionType = NeuralEnums::ActivationFunction::Tanh;
		else if (activationFunction == "GeLU")
			ActivationFunctionType = NeuralEnums::ActivationFunction::GeLU;
		else if (activationFunction == "SoftPlus")
			ActivationFunctionType = NeuralEnums::ActivationFunction::SoftPlus;
		else if (activationFunction == "SoftSign")
			ActivationFunctionType = NeuralEnums::ActivationFunction::SoftSign;
		else
			ActivationFunctionType = NeuralEnums::ActivationFunction::None;

		if (layerType == "HiddenLayer")
			LayerType = NeuralEnums::LayerType::HiddenLayer;
		else if (layerType == "InputLayer")
			LayerType = NeuralEnums::LayerType::InputLayer;
		else if (layerType == "OutputLayer")
			LayerType = NeuralEnums::LayerType::OutputLayer;
		else
			LayerType = NeuralEnums::LayerType::None;

		auto l = new Layer(size, LayerType, ActivationFunctionType, bias, DropuOutSize, neuralNetwork.BatchSize);
		neuralNetwork.Layers[counter] = *l;
		counter++;
	}
	neuralNetwork.NeuralNetworkInit();
	neuralNetwork.InitializeWeights();
}
void ReadData(std::vector<std::vector<float>>& trainingSet, std::vector<std::vector<float>>& testSet, std::vector<std::vector<float>>& labels, std::vector<std::vector<float>>& testLabels)
{

	std::vector<size_t> _labels;
	std::vector<size_t> _testlabels;
	ReadMNISTMod(trainingSet, _labels, true);
	ReadMNISTMod(testSet, _testlabels, false);

	int minmax[2];
	size_t totaltrain = trainingSet.size();
	labels.resize(totaltrain);
	for (size_t i = 0; i < totaltrain; i++)
	{
		Compress(trainingSet[i].data(), trainingSet[i].size(), minmax);
		labels[i].resize(10);
		for (size_t k = 0; k < 10; k++)
			labels[i][k] = (_labels[i] == k ? 1.0f : 0.0f);
	}

	size_t totalTest = testSet.size();
	testLabels.resize(totalTest);
	for (size_t k = 0; k < totalTest; k++)
	{
		Compress(testSet[k].data(), testSet[k].size(), minmax);
		testLabels[k].resize(10);
		for (size_t g = 0; g < 10; g++)
			testLabels[k][g] = (_testlabels[k] == g ? 1.0f : 0.0f);
	}
}
void copyNetrworkCuda(NeuralNetwork& nn, MnistData* trainingSet, int trainingSetSize, MnistData* testSet, int testSetSize, bool copyData);
int copyClass();
void readDataAndTest()
{

	//copyClass<float>();
	NeuralNetwork neuralNetwork;
	MnistData* trainingSet;
	MnistData* testSet;
	std::vector<float> losses;

	initNeUnetFromJson(neuralNetwork);
	cout << "start reading training+test data " << endl;

	clock_t begin = clock();
	size_t trainingSetSize = 0;
	size_t testSetSize = 0;

	std::vector<std::vector<float>> _trainingSet;
	std::vector<std::vector<float>> _testSet;
	std::vector<std::vector<float>> labeledTarget;
	std::vector<std::vector<float>> testLabeledTarget;
	ReadData(_trainingSet, _testSet, labeledTarget, testLabeledTarget);
	std::vector<int> index;
	trainingSet = new MnistData[_trainingSet.size()];
	index.resize(_trainingSet.size());
	for (size_t i = _trainingSet.size(); i--;)
	{
		index[i] = i;
	}
	testSet = new MnistData[_testSet.size()];
	trainingSetSize = _trainingSet.size();
	testSetSize = _testSet.size();
	for (size_t i = 0; i < _trainingSet.size(); i++)
	{
		trainingSet[i].set = new float[_trainingSet[i].size()];
		trainingSet[i].setSize = _trainingSet[i].size();
		trainingSet[i].label = new float[labeledTarget[i].size()];
		trainingSet[i].set = _trainingSet[i].data();
		trainingSet[i].label = labeledTarget[i].data();
		trainingSet[i].labelSize = labeledTarget[i].size();
	}

	for (size_t i = 0; i < _testSet.size(); i++)
	{
		testSet[i].set = new float[_testSet[i].size()];
		testSet[i].setSize = _testSet[i].size();
		testSet[i].label = new float[testLabeledTarget[i].size()];
		testSet[i].set = _testSet[i].data();
		testSet[i].label = testLabeledTarget[i].data();
		testSet[i].labelSize = testLabeledTarget[i].size();
	}

	if (neuralNetwork.Cuda)
	{
		cout << "...done" << endl;
		int deviceCount = 0;
		int cudaDevice = 0;
		char cudaDeviceName[100];
		cuInit(0);
		cuDeviceGetCount(&deviceCount);
		cuDeviceGet(&cudaDevice, 0);
		cuDeviceGetName(cudaDeviceName, 100, cudaDevice);
		cout << "found CUDA device: " + std::string(cudaDeviceName) + ". Count: " + std::to_string(deviceCount) << endl;
		copyNetrworkCuda(neuralNetwork, trainingSet, trainingSetSize, testSet, testSetSize, true);
		//copyClass();
	}
	else
	{
		clock_t end = clock();
		cout << "...done in " + std::to_string(double(end - begin) / CLOCKS_PER_SEC) + " seconds" << endl;
		size_t total = _trainingSet.size();

		cout << "start training" << endl;
		begin = clock();
		size_t globalEpochs = 300;
		size_t totalcounter = 0;
		float loss = 0;
		for (size_t g = 0; g < globalEpochs; g++)
		{
			try
			{
				size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
				//shuffle(index.begin(), index.end(), std::default_random_engine(seed));
				clock_t beginInside = clock();

				if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
				{
					for (size_t i = 0; i < total / neuralNetwork.BatchSize; i++)
					{
						if (neuralNetwork.BatchSize == 1)
						{
							neuralNetwork.Layers[0].Outputs = trainingSet[index[i]].set;
							if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
								neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = trainingSet[index[i]].label;
							else
								neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = trainingSet[index[i]].set;
						}
						else
						{
							for (size_t batch = 0; batch < neuralNetwork.BatchSize; batch++)
							{
								neuralNetwork.Layers[0].OutputsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
								if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
									neuralNetwork.Layers[neuralNetwork.LayersSize - 1].TargetsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].label;
								else
									neuralNetwork.Layers[neuralNetwork.LayersSize - 1].TargetsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
							}
						}

						loss = neuralNetwork.PropagateForwardThreaded(true, false);
						neuralNetwork.PropagateBackThreaded();
						losses.push_back(loss);

						if (neuralNetwork.BatchSize == 1)
						{
							if (i % 1000 == 0 && i != 0)
							{
								totalcounter += 1000;
								cout << std::to_string(totalcounter) + "/" + std::to_string(total * globalEpochs) << "\r";
							}
						}
						else
						{
							totalcounter++;
							if (i % 100 == 0 && i != 0)
								cout << std::to_string(totalcounter) + "/" + std::to_string(total * globalEpochs / neuralNetwork.BatchSize) << "\r";
						}
					}
				}
				if (neuralNetwork.Type == NeuralEnums::NetworkType::AutoEncoder)
				{
					for (size_t i = 0; i < total / neuralNetwork.BatchSize; i++)
					{
						if (neuralNetwork.BatchSize == 1)
						{
							neuralNetwork.Layers[0].Outputs = trainingSet[index[i]].set;
							if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
								neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = trainingSet[index[i]].label;
							else
								neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = trainingSet[index[i]].set;
						}
						else
						{
							for (size_t batch = 0; batch < neuralNetwork.BatchSize; batch++)
							{
								neuralNetwork.Layers[0].OutputsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
								if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
									neuralNetwork.Layers[neuralNetwork.LayersSize - 1].TargetsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].label;
								else
									neuralNetwork.Layers[neuralNetwork.LayersSize - 1].TargetsBatch[batch] = trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
							}
						}

						//main learning sequence
						neuralNetwork.PropagateForwardThreaded(true, true);
						//losses.push_back(loss);
						if (neuralNetwork.BatchSize == 1)
						{
							if (i % 1000 == 0 && i != 0)
							{
								totalcounter += 1000;

								cout << std::to_string(totalcounter) + "/" + std::to_string(total * globalEpochs) << "\r";
							}
						}
						else
						{
							totalcounter++;
							if (i % 100 == 0 && i != 0)
								cout << std::to_string(totalcounter) + "/" + std::to_string(total * globalEpochs / neuralNetwork.BatchSize) << "\r";
						}
					}
					//// rohat average
					//for (size_t l = 1; l < neuralNetwork.Layers.size(); l++)
					//{
					//	for (size_t f = 0; f < neuralNetwork.Layers[l].RoHat.size(); f++)
					//	{
					//		neuralNetwork.Layers[l].RoHat[f] /= total;
					//	}
					//}
				}
				clock_t endInside = clock();

				size_t counter = 0;
				size_t digitCounter = 0;
				cout << std::to_string(g + 1) + " of " + std::to_string(globalEpochs) + " done in " + std::to_string(float(endInside - beginInside) / CLOCKS_PER_SEC) + " seconds. " + ". loss: " + std::to_string(losses.size() > 0 ? losses[losses.size() - 1] : 0) << endl;

				neuralNetwork.PrepareForTesting();
				float result = 0;
				if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
				{
					if (neuralNetwork.Metrics == NeuralEnums::Metrics::Full)
					{
						beginInside = clock();
						for (size_t i = 0; i < trainingSetSize; i++)
						{
							neuralNetwork.Layers[0].Outputs = trainingSet[i].set;
							neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = trainingSet[i].label;
							auto loss = neuralNetwork.PropagateForwardThreaded(false, false);
							if (GetMaxIndex(neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Outputs, neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Size) == GetMaxIndex(trainingSet[i].label, trainingSet[i].labelSize))
								counter++;
							digitCounter++;
						}
						result = (float)counter / (float)digitCounter;
						auto testComplete = "training-set result: " + std::to_string(result);
						endInside = clock();
						cout << "...training set testing done in " + std::to_string(float(endInside - beginInside) / CLOCKS_PER_SEC) + " seconds. Result: " + std::to_string(result) << endl;
					}
					if (neuralNetwork.Metrics == NeuralEnums::Metrics::TestSet || neuralNetwork.Metrics == NeuralEnums::Metrics::Full)
					{
						beginInside = clock();
						counter = 0;
						digitCounter = 0;
						for (size_t i = 0; i < testSetSize; i++)
						{
							neuralNetwork.Layers[0].Outputs = testSet[i].set;
							neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target = testSet[i].label;
							auto loss = neuralNetwork.PropagateForwardThreaded(false, false);
							if (GetMaxIndex(neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Outputs, neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Size) == GetMaxIndex(testSet[i].label, (size_t)10))
								counter++;
							digitCounter++;
						}
						result = (float)counter / (float)digitCounter;
						auto testComplete2 = "; test-set result: " + std::to_string(result);
						endInside = clock();
						cout << "...testing set testing done in " + std::to_string(float(endInside - beginInside) / CLOCKS_PER_SEC) + " seconds. Result: " + std::to_string(result) + ". loss: " + std::to_string(losses[losses.size() - 2]) << endl;
					}
				}
				if (neuralNetwork.LogLoss == NeuralEnums::LogLoss::Full || neuralNetwork.LogLoss == NeuralEnums::LogLoss::Sparce)
				{
					std::ofstream oData;
					oData.open("loss.txt");
					for (size_t count = 0; count < losses.size(); count++)
					{
						if (neuralNetwork.LogLoss == NeuralEnums::LogLoss::Sparce && count % 10 == 0)
							oData << std::setprecision(100) << losses[count] << endl;
						else
							oData << std::setprecision(100) << losses[count] << endl;
					}
				}
				losses.clear();
			}
			catch (std::exception e)
			{
				cout << e.what() << endl;
			}
		}

		end = clock();
		cout << "training done in " + std::to_string(float(end - begin) / CLOCKS_PER_SEC) + " seconds" << endl;
	}
}

int main()
{
	std::thread test(readDataAndTest);
	//std::thread test(test);
	test.join();
	return 0;
}
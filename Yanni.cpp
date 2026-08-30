#include "Yanni.h"
#include "cliOptions.h"
#include "logger.h"
#include <ctime>
#include <sstream>

using namespace std;

void testNet() {
  float *ar = new float[4];
  // bias test case
  NeuralNetwork neuralNetwork;
  ar[0] = 1.0f;
  ar[1] = 0.9f;
  ar[2] = 0.4f;
  ar[3] = 0.7f;
  std::vector<float> labels;
  float *targetArray;
  std::vector<float> losses;
  targetArray = new float[2];
  targetArray[0] = 1.0f;
  targetArray[1] = 0.01f;
  Layer layerInput(4, NeuralEnums::LayerType::InputLayer,
                   NeuralEnums::ActivationFunction::None, 1, 0);
  Layer layerHidden1(4, NeuralEnums::LayerType::HiddenLayer,
                     NeuralEnums::ActivationFunction::Sigmoid, 1, 0);
  Layer layerHidden2(3, NeuralEnums::LayerType::HiddenLayer,
                     NeuralEnums::ActivationFunction::Sigmoid, 1, 0);
  Layer layerOutput(2, NeuralEnums::LayerType::OutputLayer,
                    NeuralEnums::ActivationFunction::Sigmoid, 0, 0);
  Layer *vecs;
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
  neuralNetwork.LossFunctionType =
      NeuralEnums::LossFunctionType::MeanSquaredError;
  neuralNetwork.LossCalculation = NeuralEnums::LossCalculation::Full;
  neuralNetwork.LayersSize = 4;
  // neuralNetwork.Layers[1].GradientsBatch.resize(1);
  // neuralNetwork.Layers[1].GradientsBatch[0].resize(12);
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
  // neuralNetwork.Layers[2].GradientsBatch[0].resize(8);
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
  // neuralNetwork.Layers[3].GradientsBatch[0].resize(6);
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

  MnistData *trainingSet;
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

  // initTrainingAndTestData<double>(neuralNetwork, trainingSet, 1, NULL, 0);
  neuralNetwork.PrepareForTesting();

  // copyNetrworkCuda(neuralNetwork, nullptr, 0, nullptr, 0, false);

  for (size_t i = 0; i < 100; i++) {
    auto loss = neuralNetwork.PropagateForwardThreaded(
        true, false); // პირველი loss უნდა იყოს 0.20739494219121993   float ზე
                      // 0.414789855
    neuralNetwork.PropagateBackThreaded();

    // 1.0002170802724326
    // 0.60019537224518926
    // 0.70008683210897293
    //- 0.39984804380929723
    // 0.99984718866787936
    //- 0.80013753019890865
    // 0.39993887546715173
    // 0.099893032067515528
    // 0.99955724383635436
    // 0.22960151945271889
    // 0.16982289753454174
    // 0.15969007068544802
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
// კონფიგის სტრიქონს enum-ად აქცევს.
//
// ადრე თითოეული ველი if-ების ჯაჭვით იპარსებოდა, else-ის გარეშე: უცნობი
// სტრიქონი ველს საერთოდ არ ანიჭებდა და იქ კონსტრუქტორის ნაგავი რჩებოდა,
// ან ჩუმად None ხდებოდა. ბრძანების ველიდან ერთი შეცდომილი ასო საკმარისი
// იყო, რომ ქსელი უაქტივაციოდ დარჩენილიყო.
template <class E>
static E ParseEnum(const char *field, const std::string &text,
                   std::initializer_list<std::pair<const char *, E>> table) {
  for (const auto &kv : table)
    if (text == kv.first)
      return kv.second;
  printf("!! %s: unknown value '%s'\n   valid: ", field, text.c_str());
  for (const auto &kv : table)
    printf("%s ", kv.first);
  printf("\n");
  exit(1);
}

static std::string JsonStr(const nlohmann::json &j, const char *key,
                           const char *fallback) {
  if (!j.contains(key) || j[key].is_null())
    return fallback;
  return j[key].get<std::string>();
}

void initNeUnetFromJson(NeuralNetwork &neuralNetwork) {
  std::ifstream ifs("netConfig.json");
  if (!ifs.is_open())
    return;

  std::string content((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));
  nlohmann::json json = nlohmann::json::parse(content);

  // ბრძანების ველი JSON-ს გადააფარებს; ფაილი უცვლელი რჩება
  if (!ApplyOverrides(json, gCli))
    exit(1);
  gEffectiveConfig = json.dump(4);

  neuralNetwork.ThreadCount = (size_t)json["ThreadCount"];
  neuralNetwork.LearningRate = json["LearningRate"];

  using AE = NeuralEnums::AutoEncoderType;
  using BT = NeuralEnums::BalanceType;
  using GT = NeuralEnums::GradientType;
  using LC = NeuralEnums::LossCalculation;
  using LL = NeuralEnums::LogLoss;
  using LF = NeuralEnums::LossFunctionType;
  using LR = NeuralEnums::LearningRateType;
  using MT = NeuralEnums::Metrics;
  using NT = NeuralEnums::NetworkType;

  neuralNetwork.BatchSize = (int)json["BatchSize"];
  neuralNetwork.Cuda = (bool)json["Cuda"];

  neuralNetwork.Type = ParseEnum<NT>("Type", JsonStr(json, "Type", "Normal"),
                                     {{"Normal", NT::Normal},
                                      {"AutoEncoder", NT::AutoEncoder}});

  neuralNetwork.Metrics =
      ParseEnum<MT>("Metrics", JsonStr(json, "Metrics", "None"),
                    {{"None", MT::None},
                     {"Test", MT::TestSet},
                     {"Full", MT::Full}});

  neuralNetwork.LogLoss =
      ParseEnum<LL>("LogLoss", JsonStr(json, "LogLoss", "None"),
                    {{"None", LL::None},
                     {"Sparce", LL::Sparce},
                     {"Full", LL::Full}});

  // ადრე else მხოლოდ ბოლო if-ს ეკვროდა, ე.ი. Contractive, Denoising და
  // Sparce დაყენების მიუხედავად მაშინვე None-ით გადაიწერებოდა
  neuralNetwork.AutoEncoderType =
      ParseEnum<AE>("AutoEncoderType", JsonStr(json, "AutoEncoderType", "None"),
                    {{"UnderComplete", AE::UnderComplete},
                     {"Sparce", AE::Sparce},
                     {"Denoising", AE::Denoising},
                     {"Contractive", AE::Contractive},
                     {"Variational", AE::Variational},
                     {"None", AE::None}});

  neuralNetwork.LossCalculation =
      ParseEnum<LC>("LossCalculation", JsonStr(json, "LossCalculation", "None"),
                    {{"None", LC::None}, {"Full", LC::Full}});

  // Cyclic ადრე პატარა ასოთი ("cyclic") იწერებოდა, ე.ი. enum-ის სახელით
  // მითითება არ მუშაობდა; AdaMax, AMSGrad და Nadam საერთოდ არ იპარსებოდა
  neuralNetwork.LearningRateType = ParseEnum<LR>(
      "LearningRateType", JsonStr(json, "LearningRateType", "Static"),
      {{"Static", LR::Static},
       {"AdaDelta", LR::AdaDelta},
       {"AdaGrad", LR::AdaGrad},
       {"Adam", LR::Adam},
       {"AdamMod", LR::AdamMod},
       {"AdaMax", LR::AdaMax},
       {"AMSGrad", LR::AMSGrad},
       {"Cyclic", LR::Cyclic},
       {"Nadam", LR::Nadam},
       {"RMSProp", LR::RMSProp}});

  neuralNetwork.BalanceType = ParseEnum<BT>(
      "Balance", JsonStr(json, "Balance", "None"),
      {{"None", BT::None},
       {"GaussianStandartization", BT::GaussianStandartization},
       {"Normalization", BT::Normalization},
       {"NormalDistrubution", BT::NormalDistrubution}});

  // მხოლოდ ეს სამია რეალიზებული lossFunctions.cpp-ში; enum-ის დანარჩენი
  // წევრები DifferentiateLossWith-ში default-ზე გადიან და ნულს აბრუნებენ,
  // ანუ ქსელი საერთოდ ვერ ისწავლიდა
  neuralNetwork.LossFunctionType = ParseEnum<LF>(
      "LossFunction", JsonStr(json, "LossFunction", "MeanSquaredError"),
      {{"MeanSquaredError", LF::MeanSquaredError},
       {"BinaryCrossentropy", LF::BinaryCrossentropy},
       {"KullbackLeiblerDivergence", LF::KullbackLeiblerDivergence}});

  neuralNetwork.GradientType =
      ParseEnum<GT>("Gradient", JsonStr(json, "Gradient", "Static"),
                    {{"Static", GT::Static}, {"Momentum", GT::Momentum}});

  neuralNetwork.Layers = new Layer[json["Layers"].size()];
  size_t counter = 0;
  neuralNetwork.LayersSize = json["Layers"].size();
  for (auto &layer : json["Layers"]) {
    using AF = NeuralEnums::ActivationFunction;
    using LT = NeuralEnums::LayerType;

    const float DropuOutSize = layer["DropuOutSize"];
    const float bias = layer["Bias"];
    const size_t size = layer["Size"];

    // ველის სახელს ლეიერის ნომერს ვამატებ, რომ შეცდომა მაშინვე იპოვებოდეს
    char field[48];
    snprintf(field, sizeof field, "layer %zu ActivationFunction", counter);
    const NeuralEnums::ActivationFunction ActivationFunctionType =
        ParseEnum<AF>(field, JsonStr(layer, "ActivationFunction", "None"),
                      {{"None", AF::None},
                       {"Sigmoid", AF::Sigmoid},
                       {"Tanh", AF::Tanh},
                       {"ReLU", AF::ReLU},
                       {"MReLU", AF::MReLU},
                       {"SoftMax", AF::SoftMax},
                       {"GeLU", AF::GeLU},
                       {"SoftPlus", AF::SoftPlus},
                       {"SoftSign", AF::SoftSign}});

    snprintf(field, sizeof field, "layer %zu Type", counter);
    const NeuralEnums::LayerType LayerType =
        ParseEnum<LT>(field, JsonStr(layer, "Type", "HiddenLayer"),
                      {{"InputLayer", LT::InputLayer},
                       {"HiddenLayer", LT::HiddenLayer},
                       {"OutputLayer", LT::OutputLayer},
                       {"None", LT::None}});

    auto l = new Layer(size, LayerType, ActivationFunctionType, bias,
                       DropuOutSize, neuralNetwork.BatchSize);
    neuralNetwork.Layers[counter] = *l;
    counter++;
  }
  neuralNetwork.NeuralNetworkInit();
  neuralNetwork.InitializeWeights();
}
void ReadData(std::vector<std::vector<float>> &trainingSet,
              std::vector<std::vector<float>> &testSet,
              std::vector<std::vector<float>> &labels,
              std::vector<std::vector<float>> &testLabels,
              const DataSetInfo &ds) {

  std::vector<int> _labels;
  std::vector<int> _testlabels;
  ReadMNISTMod(trainingSet, _labels, true, ds);
  ReadMNISTMod(testSet, _testlabels, false, ds);

  if (trainingSet.size() == 0) {
    return;
  }

  // EMNIST letters ლეიბლებს 1-იდან ითვლის, დანარჩენები 0-იდან
  const int shift = (ds.name == "EMNIST letters") ? 1 : 0;
  const size_t classes = (size_t)ds.classCount;

  int minmax[2];
  size_t totaltrain = trainingSet.size();
  labels.resize(totaltrain);
  for (size_t i = 0; i < totaltrain; i++) {
    Compress(trainingSet[i].data(), trainingSet[i].size(), minmax);
    labels[i].resize(classes);
    for (size_t k = 0; k < classes; k++)
      labels[i][k] = ((size_t)(_labels[i] - shift) == k ? 1.0f : 0.0f);
  }

  size_t totalTest = testSet.size();
  testLabels.resize(totalTest);
  for (size_t k = 0; k < totalTest; k++) {
    Compress(testSet[k].data(), testSet[k].size(), minmax);
    testLabels[k].resize(classes);
    for (size_t g = 0; g < classes; g++)
      testLabels[k][g] = ((size_t)(_testlabels[k] - shift) == g ? 1.0f : 0.0f);
  }
}
#ifdef USE_CUDA
void copyNetrworkCuda(NeuralNetwork &nn, MnistData *trainingSet,
                      int trainingSetSize, MnistData *testSet, int testSetSize,
                      bool copyData);
int copyClass();
#endif
void readDataAndTest() {

  // copyClass<float>();
  NeuralNetwork neuralNetwork;
  MnistData *trainingSet;
  MnistData *testSet;
  std::vector<float> losses;

  initNeUnetFromJson(neuralNetwork);

  // ქსელის ჩვენება ტრენინგის გარეშე: მონაცემების ჩატვირთვამდე ვჩერდებით,
  // რადგან ის რამდენიმე წამია და ტოპოლოგიისთვის საჭირო არაა
  // ტოპოლოგიის ჩვენებას დატასეტი არ სჭირდება -- პრომპტამდე ვჩერდებით
  if (gCli.topologyOnly) {
    PrintNetworkInfo(neuralNetwork, 0);
    return;
  }

  // დატასეტის არჩევა ცალკე ნაბიჯია: ის გაშვების გადაწყვეტილებაა,
  // არა ჩატვირთვის დეტალი
  const DataSetInfo ds = ChooseDataSet(gCli.dataset);
  if (ds.classCount == 0)
    return;

  LogOpen("yanni.log");
  {
    const std::time_t now = std::time(nullptr);
    char when[64];
    std::strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));
    LogComment("yanni training log, started %s", when);
    LogComment("dataset %s, %d classes", ds.name.c_str(), ds.classCount);

    long total = 0;
    for (int i = 1; i < neuralNetwork.LayersSize; i++)
      total += (long)neuralNetwork.Layers[i - 1].Size *
               (neuralNetwork.Layers[i].Size -
                (neuralNetwork.Layers[i].UsingBias ? 1 : 0));
    LogComment("parameters %ld, weights %.1f MB", total,
               total * sizeof(float) / 1e6);
    LogComment("");
    LogComment("--- effective configuration: netConfig.json + command line ---");
    // სტრიქონობრივად, რომ ყოველი ხაზი '#'-ით დაიწყოს და CSV-ის
    // მკითხველმა მთელი ბლოკი გამოტოვოს
    std::istringstream cfg(gEffectiveConfig);
    std::string line;
    while (std::getline(cfg, line))
      LogComment("%s", line.c_str());
    LogComment("");
  }
  LogLine("epoch,train_seconds,train_loss,trainset_seconds,trainset_accuracy,"
          "testset_seconds,testset_accuracy");

  // ასოებზე გადასვლისას ყველაზე ადვილი დასაშვები შეცდომა: netConfig.json-ში
  // გამომავალი ლეიერი 10 დარჩა, დატასეტს კი 26 კლასი აქვს
  const int outSize = neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Size;
  if (outSize != ds.classCount)
    printf("\n  !! output layer has %d neurons but %s has %d classes\n"
           "     set \"Size\": %d on the output layer in netConfig.json,\n"
           "     or pass --set %d.Size=%d\n\n",
           outSize, ds.name.c_str(), ds.classCount, ds.classCount,
           neuralNetwork.LayersSize - 1, ds.classCount);

  cout << "start reading training+test data " << endl;

  auto begin = std::chrono::steady_clock::now();
  size_t trainingSetSize = 0;
  size_t testSetSize = 0;

  std::vector<std::vector<float>> _trainingSet;
  std::vector<std::vector<float>> _testSet;
  std::vector<std::vector<float>> labeledTarget;
  std::vector<std::vector<float>> testLabeledTarget;

  ReadData(_trainingSet, _testSet, labeledTarget, testLabeledTarget, ds);

  if (_trainingSet.size() == 0) {
    return;
  }

  trainingSet = new MnistData[_trainingSet.size()];
  std::vector<int> index;
  index.resize(_trainingSet.size());
  for (size_t i = _trainingSet.size(); i--;) {
    index[i] = i;
  }
  testSet = new MnistData[_testSet.size()];
  trainingSetSize = _trainingSet.size();
  testSetSize = _testSet.size();
  for (size_t i = 0; i < _trainingSet.size(); i++) {
    trainingSet[i].set = new float[_trainingSet[i].size()];
    trainingSet[i].setSize = _trainingSet[i].size();
    trainingSet[i].label = new float[labeledTarget[i].size()];
    trainingSet[i].set = _trainingSet[i].data();
    trainingSet[i].label = labeledTarget[i].data();
    trainingSet[i].labelSize = labeledTarget[i].size();
  }

  for (size_t i = 0; i < _testSet.size(); i++) {
    testSet[i].set = new float[_testSet[i].size()];
    testSet[i].setSize = _testSet[i].size();
    testSet[i].label = new float[testLabeledTarget[i].size()];
    testSet[i].set = _testSet[i].data();
    testSet[i].label = testLabeledTarget[i].data();
    testSet[i].labelSize = testLabeledTarget[i].size();
  }

#ifdef USE_CUDA
  if (neuralNetwork.Cuda) {
    cout << "...done" << endl;
    int deviceCount = 0;
    int cudaDevice = 0;
    char cudaDeviceName[100];
    cuInit(0);
    cuDeviceGetCount(&deviceCount);
    cuDeviceGet(&cudaDevice, 0);
    cuDeviceGetName(cudaDeviceName, 100, cudaDevice);
    cout << "found CUDA device: " + std::string(cudaDeviceName) +
                ". Count: " + std::to_string(deviceCount)
         << endl;
    copyNetrworkCuda(neuralNetwork, trainingSet, trainingSetSize, trainingSet,
                     trainingSetSize, true);
  } else
#endif
  {
    auto end = std::chrono::steady_clock::now();
    cout << "...done in " +
                std::to_string(
                    std::chrono::duration<double>(end - begin).count()) +
                " seconds"
         << endl;
    size_t total = _trainingSet.size();

    PrintNetworkInfo(neuralNetwork, total);

    cout << "start training" << endl;
    begin = std::chrono::steady_clock::now();
    size_t globalEpochs = 300;
    size_t totalcounter = 0;
    float loss = 0;
    for (size_t g = 0; g < globalEpochs; g++) {
      try {
        size_t seed =
            std::chrono::system_clock::now().time_since_epoch().count();
        // shuffle(index.begin(), index.end(),
        // std::default_random_engine(seed));
        auto beginInside = std::chrono::steady_clock::now();

        if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal) {
          for (size_t i = 0; i < total / neuralNetwork.BatchSize; i++) {
            if (neuralNetwork.BatchSize == 1) {
              neuralNetwork.Layers[0].Outputs = trainingSet[index[i]].set;
              if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
                neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                    trainingSet[index[i]].label;
              else
                neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                    trainingSet[index[i]].set;
            } else {
              for (size_t batch = 0; batch < neuralNetwork.BatchSize; batch++) {
                neuralNetwork.Layers[0].OutputsBatch[batch] =
                    trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
                if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
                  neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                      .TargetsBatch[batch] =
                      trainingSet[index[i * neuralNetwork.BatchSize + batch]]
                          .label;
                else
                  neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                      .TargetsBatch[batch] =
                      trainingSet[index[i * neuralNetwork.BatchSize + batch]]
                          .set;
              }
            }

            loss = neuralNetwork.PropagateForwardThreaded(true, false);
            neuralNetwork.PropagateBackThreaded();
            losses.push_back(loss);

            if (neuralNetwork.BatchSize == 1) {
              if (i % 1000 == 0 && i != 0) {
                totalcounter += 1000;
                if (!gCli.quiet)
                  cout << std::to_string(totalcounter) + "/" +
                            std::to_string(total * globalEpochs)
                     << "\r";
              }
            } else {
              totalcounter++;
              if (i % 100 == 0 && i != 0)
                if (!gCli.quiet)
                  cout << std::to_string(totalcounter) + "/" +
                            std::to_string(total * globalEpochs /
                                           neuralNetwork.BatchSize)
                     << "\r";
            }
          }
        }
        if (neuralNetwork.Type == NeuralEnums::NetworkType::AutoEncoder) {
          for (size_t i = 0; i < total / neuralNetwork.BatchSize; i++) {
            if (neuralNetwork.BatchSize == 1) {
              neuralNetwork.Layers[0].Outputs = trainingSet[index[i]].set;
              if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
                neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                    trainingSet[index[i]].label;
              else
                neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                    trainingSet[index[i]].set;
            } else {
              for (size_t batch = 0; batch < neuralNetwork.BatchSize; batch++) {
                neuralNetwork.Layers[0].OutputsBatch[batch] =
                    trainingSet[index[i * neuralNetwork.BatchSize + batch]].set;
                if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal)
                  neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                      .TargetsBatch[batch] =
                      trainingSet[index[i * neuralNetwork.BatchSize + batch]]
                          .label;
                else
                  neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                      .TargetsBatch[batch] =
                      trainingSet[index[i * neuralNetwork.BatchSize + batch]]
                          .set;
              }
            }

            // main learning sequence
            neuralNetwork.PropagateForwardThreaded(true, true);
            // losses.push_back(loss);
            if (neuralNetwork.BatchSize == 1) {
              if (i % 1000 == 0 && i != 0) {
                totalcounter += 1000;

                if (!gCli.quiet)
                  cout << std::to_string(totalcounter) + "/" +
                            std::to_string(total * globalEpochs)
                     << "\r";
              }
            } else {
              totalcounter++;
              if (i % 100 == 0 && i != 0)
                if (!gCli.quiet)
                  cout << std::to_string(totalcounter) + "/" +
                            std::to_string(total * globalEpochs /
                                           neuralNetwork.BatchSize)
                     << "\r";
            }
          }
          //// rohat average
          // for (size_t l = 1; l < neuralNetwork.Layers.size(); l++)
          //{
          //	for (size_t f = 0; f < neuralNetwork.Layers[l].RoHat.size();
          // f++)
          //	{
          //		neuralNetwork.Layers[l].RoHat[f] /= total;
          //	}
          // }
        }
        auto endInside = std::chrono::steady_clock::now();

        size_t counter = 0;
        size_t digitCounter = 0;
        if (!gCli.quiet)
          cout << std::to_string(g + 1) + " of " + std::to_string(globalEpochs) +
                    " done in " +
                    std::to_string(
                        std::chrono::duration<double>(endInside - beginInside)
                            .count()) +
                    " seconds. " + ". loss: " +
                    std::to_string(losses.size() > 0 ? losses[losses.size() - 1]
                                                     : 0)
             << endl;

        // ერთი ეპოქა = ერთი სტრიქონი ლოგში; ტესტების შედეგებს ქვემოთ
        // ვაგროვებთ და სტრიქონს ბოლოში ვწერთ
        const double epochSeconds =
            std::chrono::duration<double>(endInside - beginInside).count();
        const float epochLoss =
            losses.size() > 0 ? losses[losses.size() - 1] : 0.0f;
        double trainsetSec = 0, testsetSec = 0;
        double trainsetAcc = -1, testsetAcc = -1; // -1 = არ გაზომილა

        neuralNetwork.PrepareForTesting();
        float result = 0;
        if (neuralNetwork.Type == NeuralEnums::NetworkType::Normal) {
          if (neuralNetwork.Metrics == NeuralEnums::Metrics::Full) {
            beginInside = std::chrono::steady_clock::now();
            for (size_t i = 0; i < trainingSetSize; i++) {
              neuralNetwork.Layers[0].Outputs = trainingSet[i].set;
              neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                  trainingSet[i].label;
              auto loss = neuralNetwork.PropagateForwardThreaded(false, false);
              if (GetMaxIndex(neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                                  .Outputs,
                              neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                                  .Size) ==
                  GetMaxIndex(trainingSet[i].label, trainingSet[i].labelSize))
                counter++;
              digitCounter++;
            }
            result = (float)counter / (float)digitCounter;
            auto testComplete =
                "training-set result: " + std::to_string(result);
            endInside = std::chrono::steady_clock::now();
            if (!gCli.quiet)
              cout << "...training set testing done in " +
                        std::to_string(std::chrono::duration<double>(
                                           endInside - beginInside)
                                           .count()) +
                        " seconds. Result: " + std::to_string(result)
                 << endl;
            trainsetSec =
                std::chrono::duration<double>(endInside - beginInside).count();
            trainsetAcc = result;
          }
          if (neuralNetwork.Metrics == NeuralEnums::Metrics::TestSet ||
              neuralNetwork.Metrics == NeuralEnums::Metrics::Full) {
            beginInside = std::chrono::steady_clock::now();
            counter = 0;
            digitCounter = 0;
            for (size_t i = 0; i < testSetSize; i++) {
              neuralNetwork.Layers[0].Outputs = testSet[i].set;
              neuralNetwork.Layers[neuralNetwork.LayersSize - 1].Target =
                  testSet[i].label;
              auto loss = neuralNetwork.PropagateForwardThreaded(false, false);
              if (GetMaxIndex(neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                                  .Outputs,
                              neuralNetwork.Layers[neuralNetwork.LayersSize - 1]
                                  .Size) ==
                  GetMaxIndex(testSet[i].label, testSet[i].labelSize))
                counter++;
              digitCounter++;
            }
            result = (float)counter / (float)digitCounter;
            auto testComplete2 = "; test-set result: " + std::to_string(result);
            endInside = std::chrono::steady_clock::now();
            if (!gCli.quiet)
              cout << "...testing set testing done in " +
                        std::to_string(std::chrono::duration<double>(
                                           endInside - beginInside)
                                           .count()) +
                        " seconds. Result: " + std::to_string(result) +
                        ". loss: " + std::to_string(losses[losses.size() - 2])
                 << endl;
            testsetSec =
                std::chrono::duration<double>(endInside - beginInside).count();
            testsetAcc = result;
          }
        }
        // გაზომვის გარეშე დარჩენილი სვეტი ცარიელი რჩება, არა 0 --
        // თორემ გრაფიკზე ნულოვან სიზუსტედ დაიხატებოდა
        char tsAcc[32] = "", teAcc[32] = "";
        if (trainsetAcc >= 0)
          snprintf(tsAcc, sizeof tsAcc, "%.6f", trainsetAcc);
        if (testsetAcc >= 0)
          snprintf(teAcc, sizeof teAcc, "%.6f", testsetAcc);
        LogLine("%zu,%.6f,%.8f,%.6f,%s,%.6f,%s", g + 1, epochSeconds, epochLoss,
                trainsetSec, tsAcc, testsetSec, teAcc);

        if (neuralNetwork.LogLoss == NeuralEnums::LogLoss::Full ||
            neuralNetwork.LogLoss == NeuralEnums::LogLoss::Sparce) {
          std::ofstream oData;
          oData.open("loss.txt");
          for (size_t count = 0; count < losses.size(); count++) {
            if (neuralNetwork.LogLoss == NeuralEnums::LogLoss::Sparce &&
                count % 10 == 0)
              oData << std::setprecision(100) << losses[count] << endl;
            else
              oData << std::setprecision(100) << losses[count] << endl;
          }
        }
        losses.clear();
      } catch (std::exception e) {
        cout << e.what() << endl;
      }
    }

    end = std::chrono::steady_clock::now();
    LogComment("training done in %.3f seconds",
               std::chrono::duration<double>(end - begin).count());
    LogClose();
    cout << "training done in " +
                std::to_string(
                    std::chrono::duration<double>(end - begin).count()) +
                " seconds"
         << endl;
  }
}

int main(int argc, char **argv) {
  gCli = ParseCli(argc, argv);
  if (gCli.help || gCli.bad) {
    PrintCliHelp(argv[0]);
    return gCli.bad ? 1 : 0;
  }
  srand(time(NULL));
  std::thread test(readDataAndTest);
  // std::thread test(testNet);
  test.join();
  return 0;
}

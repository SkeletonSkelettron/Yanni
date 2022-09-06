#include "../include/NeuralNetwork.h"
#include "../include/MnistData.h"

extern "C" void copyNetrworkCuda(NeuralNetwork& nn, MnistData* trainingSet, int trainingSetSize, MnistData* testSet, int testSetSize, bool notTesting);
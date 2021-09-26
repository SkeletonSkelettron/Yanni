#ifndef READMNIST_H
#define READMNIST_H


#include <string>
#include <cuchar>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
# include <windows.h>

unsigned int in(std::ifstream& icin, unsigned int size);

void ReadMNISTMod(std::vector<std::vector<float>>& images, std::vector<int>& labels, bool train);


#endif // READMNIST_H
#ifndef READMNIST_H
#define READMNIST_H


#include <string>
#include <cuchar>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
# include <windows.h>

size_t in(std::ifstream& icin, size_t size);

void ReadMNISTMod(std::vector<std::vector<float>>& images, std::vector<size_t>& labels, bool train);


#endif // READMNIST_H
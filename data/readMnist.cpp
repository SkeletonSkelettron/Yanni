#include "readMnist.h"
#include <string>
#include <fstream>
#include <cstring>
#include <vector>

unsigned int in(std::ifstream& icin, unsigned int size)
{
	unsigned int ans = 0;
	for (long int i = 0; i < size; i++)
	{
		unsigned char x;
		icin.read((char*)&x, 1);
		unsigned int temp = x;
		ans <<= 8;
		ans += temp;
	}
	return ans;
}

void ReadMNISTMod(std::vector<std::vector<float>>& images, std::vector<int>& labels, bool train)
{
	unsigned int num, magic, rows, cols;
	std::ifstream icin;
	icin.open(train ? "mnist/train-images.idx3-ubyte"
		: "mnist/t10k-images.idx3-ubyte", std::ios::binary);
	magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
	std::vector<float> img;
	std::vector<std::vector<float>> img2;
	for (long int i = 0; i < num; i++)
	{

		img.resize(rows * cols);
		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < cols; y++)
			{
				img[rows * x + y] = in(icin, 1);
			}
		}
		images.push_back(img);
		img.clear();
	}

	icin.close();
	icin.open(train ? "mnist/train-labels.idx1-ubyte"
		: "mnist/t10k-labels.idx1-ubyte", std::ios::binary);
	long int num2_ = num;
	magic = in(icin, 4), num2_ = in(icin, 4);
	for (long int i = 0; i < num; i++)
	{
		labels.push_back(in(icin, 1));
	}
}


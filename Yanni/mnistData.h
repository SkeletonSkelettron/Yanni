#pragma once
struct MnistData
{
	size_t setSize;
	size_t labelSize;
	size_t minMax[2];
	float* set;
	float* label;
};

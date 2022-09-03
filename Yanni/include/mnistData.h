#pragma once
struct MnistData
{
	int setSize;
	int labelSize;
	int minMax[2];
	float* set;
	float* label;
};

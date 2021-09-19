#ifndef ENUMS_H
#define ENUMS_H
namespace NeuralEnums {
	enum class ActivationFunction
	{
		None = 0,
		Sigmoid = 1,
		Tanh = 2,
		ReLU = 3,
		MReLU = 4,
		SoftMax = 5,
		GeLU = 6,
		SoftPlus = 7,
		SoftSign = 8
	};
	enum class LayerType
	{
		InputLayer = 0,
		HiddenLayer = 1,
		OutputLayer = 2,
		None = 3
	};
	enum class BalanceType
	{
		None = 0,
		// Mean = 0, StDev = 1
		GaussianStandartization = 1,
		// x(i)/Range
		Normalization = 2,
		//Probabilities are distributed normaly
		NormalDistrubution = 3,
	};
	enum class LossFunctionType
	{
		MeanSquaredError = 0,
		MeanAbsoluteError = 1,
		MeanAbsolutePercentageError = 2,
		MeanSquaredLogarithmicError = 3,
		SquaredHinge = 4,
		Hinge = 5,
		CategoricalHinge = 6,
		LogCosh = 7,
		CategoricalCrossentropy = 8,
		SparseCategoricalCrossEntropy = 9,
		BinaryCrossentropy = 10,
		KullbackLeiblerDivergence = 11,
		Poisson = 12,
		CosineProximity = 13,
	};

	enum class LearningRateType
	{
		AdaDelta = 0,
		AdaGrad = 1,
		Adam = 2,
		AdamMod = 3,
		AdaMax = 4,
		AMSGrad = 5,
		Cyclic = 6,
		GuraMethod = 7,
		Nadam = 8,
		RMSProp = 9,
		Static = 10,
	};

	enum class GradientType
	{
		Static = 0,
		Momentum = 1,
	};

	enum class NetworkType
	{
		Normal = 0,
		AutoEncoder = 1,
	};

	enum class Precision
	{
		Float = 0,
		Double = 1,
		LongDouble = 2,
		BoostBinFloat50 = 3,
		BoostBinFloat100 = 4,
	};

	enum class Metrics
	{
		None = 0,
		TestSet = 1,
		Full = 2,
	};

	enum class LossCalculation
	{
		None = 0,
		Full = 1,
	};

	enum class AutoEncoderType {
		UnderComplete = 0,
		Sparce = 1,
		Denoising = 2,
		Contractive = 3,
		Variational = 4,
		None = 5,
	};

	enum class LogLoss
	{
		None = 0,
		Sparce = 1,
		Full = 2,
	};
}
#endif // ENUMS_H



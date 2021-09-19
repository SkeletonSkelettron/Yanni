namespace NeuNetCuda {
	enum class ActivationFunction
	{
		None,
		Sigmoid,
		Tanh,
		ReLU,
		MReLU,
		SoftMax,
		GeLU,
		SoftPlus,
		SoftSign
	};
	enum class LayerType
	{
		InputLayer,
		HiddenLayer,
		OutputLayer
	};
	enum class BalanceType
	{
		None,
		// Mean = 0, StDev = 1
		GaussianStandartization,
		// x(i)/Range
		Normalization,
		//Probabilities are distributed normaly
		NormalDistrubution
	};
	enum class LossFunctionType
	{
		MeanSquaredError,
		MeanAbsoluteError,
		MeanAbsolutePercentageError,
		MeanSquaredLogarithmicError,
		SquaredHinge,
		Hinge,
		CategoricalHinge,
		LogCosh,
		CategoricalCrossentropy,
		SparseCategoricalCrossEntropy,
		BinaryCrossentropy,
		KullbackLeiblerDivergence,
		Poisson,
		CosineProximity
	};

	enum class LearningRateType
	{
		AdaDelta,
		AdaGrad,
		Adam,
		AdamMod,
		AdaMax,
		AMSGrad,
		Cyclic,
		GuraMethod,
		Nadam,
		RMSProp,
		Static
	};

	enum class GradientType
	{
		Static,
		Momentum
	};

	enum class NetworkType
	{
		Normal,
		AutoEncoder
	};

	enum class Precision
	{
		Float,
		Double,
		LongDouble,
		BoostBinFloat50,
		BoostBinFloat100
	};

	enum class Metrics
	{
		None,
		TestSet,
		Full
	};

	enum class LossCalculation
	{
		None,
		Full
	};

	enum class AutoEncoderType {
		UnderComplete,
		Sparce,
		Denoising,
		Contractive,
		Variational
	};

	enum class LogLoss
	{
		None,
		Sparce,
		Full
	};
}
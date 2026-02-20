//#include <math.h>
//namespace CudaNeuNet {
//	template <typename T> T GetGradient(NeuralNetworkCuda<T>& nn, T& gradient, int& j, int& iterator)
//	{
//		switch (nn.GradientType)
//		{
//		case NeuralEnums::GradientType::Static: return gradient;
//		case NeuralEnums::GradientType::Momentum: return Momentum(nn.Layers[iterator].GradientsForGrads, gradient, j);
//
//		default:
//			break;
//		}
//	}
//
//	template <typename T> T Momentum(T* gradients, T& gradient, int& j)
//	{
//		const T MomentumRate = 0.9;
//		gradients[j] = MomentumRate * gradients[j] + (1 - MomentumRate) * gradient;
//
//		return gradients[j];
//	}
//}
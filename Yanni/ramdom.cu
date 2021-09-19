//#include <iostream>
//#include <Cuda.h>
//#include<curand.h>
//#include<curand_kernel.h>
//
//
//int n = 200;
//using namespace std;
//
//__device__ float generate(curandState* globalState, int ind)
//{
//    //int ind = threadIdx.x;
//    curand_init(1234, ind, 0, &globalState[ind]);
//    curandState localState = globalState[ind];
//    
//    float RANDOM = curand_uniform(&localState);
//    globalState[ind] = localState;
//    return RANDOM;
//}
////
////__global__ void setup_kernel(curandState* state, unsigned long seed)
////{
////    int id = threadIdx.x;
////    curand_init(seed, id, 0, &state[id]);
////}
//
//__global__ void kernel(float* N, curandState* globalState, int n)
//{
//    // generate random numbers
//    for (int i = 0; i < 200; i++)
//    {
//        int k = generate(globalState, i) * 100000;
//        while (k > n * n - 1)
//        {
//            k -= (n * n - 1);
//        }
//        N[i] = k;
//    }
//}
//
//int generateRand()
//{
//    int N = 200;
//
//    curandState* devStates;
//    cudaMalloc(&devStates, N * sizeof(curandState));
//
//    //// setup seeds
//    //setup_kernel << < 1, 1 >> > (devStates, unsigned(time(NULL)));
//
//    float N2[200];
//    float* N3;
//    cudaMalloc((void**)&N3, sizeof(float) * N);
//
//    kernel << <1, 1 >> > (N3, devStates, n);
//    cudaDeviceSynchronize();
//    cudaMemcpy(N2, N3, sizeof(float) * N, cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < N; i++)
//    {
//        cout << N2[i] << endl;
//    }
//
//    return 0;
//}
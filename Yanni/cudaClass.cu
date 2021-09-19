#include <stdio.h>
#include "cuda.h"
#include "device_launch_parameters.h"
class CudaClass
{
public:
	int* data;
	int val;
	CudaClass() {}
	CudaClass(int x, int val_) 
	{
		data = new int[1]; data[0] = x;
		val = val_;
	}
};
__device__  CudaClass* cls;
 __global__ void useClass(CudaClass* cudaClass)
{
	cls = cudaClass;
	cls->val = 3;
};
int copyClass()
{
	CudaClass c(2, 2);
	// create class storage on device and copy top level class
	CudaClass* d_c;
	cudaMalloc((void**)&d_c, sizeof(CudaClass));
	cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
	// make an allocated region on device for use by pointer in class
	int* hostdata;
	cudaMalloc((void**)&hostdata, sizeof(int));
	cudaMemcpy(hostdata, c.data, sizeof(int), cudaMemcpyHostToDevice);
	// copy pointer to allocated device storage to device class
	cudaMemcpy(&(d_c->data), &hostdata, sizeof(int*), cudaMemcpyHostToDevice);
	useClass << <1, 1 >> > (d_c);
	cudaDeviceSynchronize();
	return 0;
}
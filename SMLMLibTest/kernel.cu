
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <array>
#include <tuple>

#define PALALA_CUDA
#include "palala.h"


__global__ void emptyKernel()
{
}

void cudaInit()
{
	// Calling an empty kernel will make sure all cuda DLLs have been loaded before doing any speed tests
	emptyKernel << < dim3(1), dim3(1) >> > ();

}

__global__ void kernelWithBreakPoint(int *p)
{
	*p = 1234;
}

void cudaBreak()
{
	DeviceArray<int> test(1);

	kernelWithBreakPoint << < dim3(1), dim3(1) >> > (test.data());

	assert(test.ToVector()[0] == 1234);
}



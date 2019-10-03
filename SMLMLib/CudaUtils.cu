#include "CudaUtils.h"
#include "ThreadUtils.h"

void EmptyKernel(cudaStream_t s) {
	LaunchKernel(1, [=]__device__(int i) {}, 0, s);
}

CDLL_EXPORT int CudaGetNumDevices()
{
	int c;
	cudaGetDeviceCount(&c);
	return c;
}

CDLL_EXPORT bool CudaSetDevice(int index)
{
	return cudaSetDevice(index) == cudaSuccess;
}

CDLL_EXPORT bool CudaGetDeviceInfo(int index, int& numMultiprocessors, char* name, int namelen)
{
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, index) != cudaSuccess)
		return false;

	numMultiprocessors = prop.multiProcessorCount;
	strcpy_s(name, namelen, prop.name);
	return true;
}

static std::mutex pinnedMemMutex, devicePitchedMemMutex, deviceMemMutex;
static int pinnedMemAmount=0, devicePitchedMemAmount=0, deviceMemAmount = 0, devicePitchedNumAllocs=0;

int CudaMemoryCounter::AddPinnedMemory(int amount)
{
	return LockedFunction(pinnedMemMutex, [&]() {
		pinnedMemAmount += amount;
		return pinnedMemAmount;
	});
}

int CudaMemoryCounter::AddDevicePitchedMemory(int amount)
{
	return LockedFunction(devicePitchedMemMutex, [&]() {
		if (amount > 0) devicePitchedNumAllocs++;
		else devicePitchedNumAllocs--;
		devicePitchedMemAmount += amount;
		return devicePitchedMemAmount;
	});
}

int CudaMemoryCounter::AddDeviceMemory(int amount)
{
	return LockedFunction(deviceMemMutex, [&]() {
		deviceMemAmount += amount;
		return deviceMemAmount;
	});
}

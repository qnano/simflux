#pragma once

#include <cufft.h>
#include "CudaUtils.h"

class FFTPlan2D
{
public:
	FFTPlan2D(int count, int width, int height, int pitchInElems, cufftType type = CUFFT_C2C);
	~FFTPlan2D();

	void Transform(const DeviceImage<cufftComplex>& src, DeviceImage<cufftComplex>& dst, bool forward, cudaStream_t stream = 0);
	void Transform(const cufftComplex* src, cufftComplex* dst, bool forward, cudaStream_t stream = 0);
	void TransformInPlace(DeviceImage<cufftComplex>& d, bool forward, cudaStream_t stream = 0);

	cufftHandle handle;
};

class FFTPlan1D
{
public:
	FFTPlan1D(int len, int count, bool forward);
	~FFTPlan1D();

	void Transform(const cuFloatComplex* src, cuFloatComplex* dst, cudaStream_t stream = 0);
	//void Transform(const float* src, cuFloatComplex* dst, cudaStream_t stream = 0);
	//void Transform(const cuFloatComplex* src, float* dst, cudaStream_t stream = 0);

	int OutputLength() const { return outputlen; }
	int InputLength() const { return inputlen; }
	int Direction() const { return direction; }
private:
	int direction;
	int inputlen, outputlen;
	cufftHandle handle;
};

CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward);
CDLL_EXPORT void FFT2(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int sigw, int sigh, int forward);

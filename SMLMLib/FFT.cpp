#include "MemLeakDebug.h"

#include "FFT.h"

static const char *cufftErrorString(cufftResult_t r)
{
	switch (r) {
	case CUFFT_SUCCESS: return "success";
	case CUFFT_INVALID_PLAN: return "invalid plan";
	case CUFFT_ALLOC_FAILED: return "alloc failed";
	case CUFFT_INVALID_TYPE: return "invalid type";
	case CUFFT_INVALID_VALUE: return "invalid value";
	case CUFFT_INTERNAL_ERROR: return "internal error";
	case CUFFT_EXEC_FAILED: return "exec failed";
	case CUFFT_SETUP_FAILED: return "setup failed";
	case CUFFT_INVALID_SIZE: return "invalid size";
	case CUFFT_UNALIGNED_DATA: return "unaligned data";
	default: return "unknown error";
	}
}

static void ThrowIfError(cufftResult_t r)
{
	if (r != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("CUFFT Error: %s", cufftErrorString(r)));
}

FFTPlan2D::FFTPlan2D(int count, int width, int height, int pitchInElems, cufftType type)
{
	int n[2] = { height,width };
	int rank = 2;
	int inembed[2] = { height, pitchInElems };
	int idist = height * pitchInElems; // size of a single complex image
	int istride = 1;
	int onembed[2] = { height, pitchInElems };
	int ostride = 1;
	int odist = height * pitchInElems;
	cufftResult result = cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, count);
	if (result != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("cufftPlanMany failed: %s", cufftErrorString(result)));

	size_t workAreaSize;
	cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, count, &workAreaSize);
#ifdef _DEBUG
	DebugPrintf("work area size for %d x %d x %d fft: %d bytes\n", count, height, width, workAreaSize);
#endif
}

FFTPlan2D::~FFTPlan2D()
{
	if (handle != 0)
		cufftDestroy(handle);
	handle = 0;
}

void FFTPlan2D::Transform(const cufftComplex* src, cufftComplex* dst, bool forward, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, (cufftComplex*)src, dst, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

void FFTPlan2D::Transform(const DeviceImage<cufftComplex>& src, DeviceImage<cufftComplex>& dst, bool forward, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, src.data, dst.data, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

void FFTPlan2D::TransformInPlace(DeviceImage<cufftComplex>& d, bool forward, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, d.data, d.data, forward?  CUFFT_FORWARD : CUFFT_INVERSE));
}

FFTPlan1D::~FFTPlan1D()
{
	if (handle != 0)
		cufftDestroy(handle);
	handle = 0;
}

FFTPlan1D::FFTPlan1D(int len, int count, bool forward)
{	
	// According to 
	//https://docs.nvidia.com/cuda/cufft/index.html
	//  1D layout is:
	// input[b * idist + x * istride]
	// output[b * odist + x * ostride]
	int n[1] = { len };
	int rank = 1;
	int idist = len; // distance between batches
	int istride = 1; // distance between elements
	int ostride = 1;
	int outputlen = len;
	int odist = len; // distance between output batches
	cufftResult result = cufftPlanMany(&handle, rank, n, 0, istride, idist, 0, ostride, odist, CUFFT_C2C, count);
	if (result != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("cufftPlanMany failed: %s", cufftErrorString(result)));

	direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
	outputlen = len;
	inputlen = len;
}

void FFTPlan1D::Transform(const cuFloatComplex* src, cuFloatComplex* dst, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, (cufftComplex*)src, dst, direction));
}



CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)
{
	FFTPlan1D plan(siglen, batchsize, !!forward);

	DeviceArray<cuFloatComplex> d_src(siglen*batchsize, src);
	DeviceArray<cuFloatComplex> d_dst(siglen*batchsize);

	plan.Transform(d_src.ptr(), d_dst.ptr());

	d_dst.CopyToHost(dst, false);
	if (!forward) {
		// CUDA FFT needs rescaling to match numpy's fft/ifft. 
		float f = 1.0f / siglen;
		for (int i = 0; i < batchsize; i++)
			for (int j = 0; j < siglen; j++) {
				dst[siglen*i + j] *= f;
			}
	}
}


CDLL_EXPORT void FFT2(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int sigw, int sigh, int forward)
{
	FFTPlan2D plan(batchsize, sigw, sigh, sigw);

	DeviceArray<cuFloatComplex> d_src(sigw*sigh*batchsize, src);
	DeviceArray<cuFloatComplex> d_dst(sigw*sigh*batchsize);

	plan.Transform(d_src.ptr(), d_dst.ptr(), forward);

	d_dst.CopyToHost(dst, false);
	if (!forward) {
		// CUDA FFT needs rescaling to match numpy's fft/ifft. 
		float f = 1.0f / (sigw*sigh);
		for (int i = 0; i < batchsize; i++)
			for (int j = 0; j < sigw*sigh; j++) {
				dst[sigw*sigh*i + j] *= f;
			}
	}
}



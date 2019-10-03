/*

GaussianPSF_XYIBg:		Sigma
GaussianPSF_XYIBgSigma	No calib.
GaussianPSF_XYZIBg		Z calibration
CubicSplinePSF			Spline coefficients
BSplinePSF				Spline coefficients


CalibrationData
PSFImpl< Model, Calibration >

*/
#pragma once

#include <driver_types.h>
#include "DLLMacros.h"
#include "Vector.h"
#include "ContainerUtils.h"
#include "CudaUtils.h"
#include <unordered_map>
#include "Context.h"

template<typename T>
class DeviceArray;


class PSFBase : public ContextObject
{
	std::string thetaFormat; 
	std::vector<int> sampleSize; // for example: [frames, height, width] or [height, width]
	int sampleCount; // = product(sampleSize)
	int numConstants;
	int thetaSize;
	int diagsize; // number of float in diagnostics / debug info per spot
public:
	DLL_EXPORT PSFBase(const std::vector<int>& sampleSize, int numConst, int thetaSize, int diagsize, const char* thetaFormat, Context* ctx=0);
	virtual ~PSFBase();

	DLL_EXPORT int IndexInTheta(const char *name);
	const char* ThetaFormat() { return thetaFormat.c_str(); }// Comma separated like "x,y,I,bg
	int ThetaSize() { return thetaSize;  }
	int SampleCount() { return sampleCount; }
	int NumConstants() { return numConstants; } // Parameters that vary per spot, but are not estimated (like roix,roiy)
	int SampleIndexDims() {	return (int)sampleSize.size(); } // Number of dimensions for a sample index (x,y or frame,x,y)
	int SampleSize(int dim) { return sampleSize[dim]; } // 0 <= dim < SampleIndexDims()
	int DiagSize() { return diagsize; }
	const std::vector<int>& SampleSize() { return sampleSize; }
};

class CUDA_PSF;

// Abstract PSF Model. All pointers point to host memory
class PSF : public PSFBase
{
public:
	PSF(const std::vector<int>& sampleSize, int numConst, int thetaSize, int diagsize, const char* thetaFormat) :
		PSFBase(sampleSize, numConst, thetaSize, diagsize, thetaFormat) {}

	// Return CUDA PSF if this is a CUDA_PSF_Wrapper
	virtual CUDA_PSF* GetCUDA_PSF() { return 0; }

	virtual void FisherMatrix(const float* theta, const float* h_const, const int* spot_pos, int numspots, float* d_FI) = 0;

	// d_image[numspots, SampleCount()], d_theta[numspots, ThetaSize()]
	virtual void ExpectedValue(float* expectedvalue, const float* theta, const float* _const, const int* spot_pos, int numspots) = 0;
	// d_deriv[numspots, ThetaSize(), SampleCount()], d_expectedvalue[numspots, SampleCount()], d_theta[numspots, ThetaSize()]
	virtual void Derivatives(float* deriv, float *expectedvalue, const float* theta, const float* _const, const int* spot_pos, int numspots) = 0;

	// d_sample[numspots, SampleCount()], d_initial[numspots, ThetaSize()], d_theta[numspots, ThetaSize()], d_iterations[numspots]
	virtual void Estimate(const float* sample, const float* d_const, const int* spot_pos, 
		const float* initial, float* theta, float* diagnostics, int* iterations, int numspots, float * trace, int traceBufLen) = 0;

	DLL_EXPORT void FisherToCRLB(const float *h_fi, float* h_crlb, int numspots);

	template<typename T>
	struct VectorType {
		typedef std::vector<T> type;
	};
};


// Abstract PSF Model. All pointers with d_ prefix point to cuda memory. 
// Default implementation uses Derivatives() to implement Estimate() and FisherMatrix()
class CUDA_PSF : public PSFBase
{
public:
	CUDA_PSF(const std::vector<int>& sampleSize, int numConst, int thetaSize, int diagsize, int maxIterations, const char* thetaFormat);
	~CUDA_PSF();

	virtual void FisherMatrix(const float* d_theta, const float* d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream);
	// d_image[numspots, SampleCount()], d_theta[numspots, ThetaSize()]
	virtual void ExpectedValue(float* expectedvalue, const float* d_theta, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) = 0;
	// d_deriv[numspots, ThetaSize(), SampleCount()], d_expectedvalue[numspots, SampleCount()], d_theta[numspots, ThetaSize()]
	// derivatives output format: [numspots, ThetaSize(), SampleCount()]
	virtual void Derivatives(float* deriv, float *expectedvalue, const float* theta, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) = 0;

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, ThetaSize()], d_theta[numspots, ThetaSize()], d_iterations[numspots]
	virtual void Estimate(const float* d_sample, const float* d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diagnostics, int* iterations,
		int numspots, float * trace, int traceBufLen, cudaStream_t stream);

//	virtual void InitialEstimate(const float* d_sample, const float* d_const, const int* d_roipos, const float* d_initial, float* d_theta, int numspots) = 0;

	DLL_EXPORT void FisherToCRLB(const float *d_fi, float* d_crlb, int numspots, cudaStream_t stream);

	template<typename T>
	struct VectorType {
		typedef DeviceArray<T> type;
	};

	virtual void SetMaxSpots(int maxspots);
	int MaxIterations() { return maxiterations;  }

protected:

	class DeviceBuffers
	{
	public:
		DeviceBuffers(int smpcount, int numspots, int thetasize);
		~DeviceBuffers();
		DeviceArray<float> derivatives, expectedValue;
	};

	int maxspots, maxiterations;

	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;
	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots);
};

CDLL_EXPORT PSF* PSF_WrapCUDA_PSF(CUDA_PSF* cuda_psf);

// Ignore the psf->Derivatives and implement Derivatives(), FisherMatrix() and Estimate() using numerically computed derivatives
// Epsilon has size [ThetaSize()] and sets the numerical derivative step size for each theta element.
CDLL_EXPORT CUDA_PSF* CUDA_PSF_NumericalDerivatives(CUDA_PSF* psf, const float* epsilon);

// C/Python API - All pointers are host memory
CDLL_EXPORT void PSF_Delete(PSF* psf);
CDLL_EXPORT const char *PSF_ThetaFormat(PSF* psf);
CDLL_EXPORT int PSF_ThetaSize(PSF* psf);
CDLL_EXPORT int PSF_SampleCount(PSF* psf);
CDLL_EXPORT int PSF_NumConstants(PSF* psf);
CDLL_EXPORT int PSF_NumDiag(PSF* psf);
CDLL_EXPORT void PSF_ComputeExpectedValue(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* ev);
//CDLL_EXPORT void PSF_ComputeInitialEstimate(PSF* psf, int numspots, const float* sample, const float* constants, float* theta);
CDLL_EXPORT void PSF_ComputeMLE(PSF* psf, int numspots, const float* sample, const float* constants, const int* spotpos, const float* initial, float* theta,
	float* diagnostics, int* iterations, float* trace, int traceBufLen);
CDLL_EXPORT void PSF_ComputeFisherMatrix(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* fi);
CDLL_EXPORT void PSF_ComputeDerivatives(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* derivatives, float* ev);

// Utils that use the PSF objects
CDLL_EXPORT void PSF_DrawSpots(PSF * psf, const float * theta, const float* constants, const Int2* roipos, int numspots);


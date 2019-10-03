#pragma once

#include "DLLMacros.h"

#include <vector>
#include "Vector.h"
#include "MathUtils.h"
#include "Estimation.h"
#include "PSFModels/Gaussian/GaussianPSF.h"


struct SIMFLUX_ASW_Params {
	int roisize;
	int numepp;
	float psfSigma;
	int maxLevMarIterations;
	float levMarLambdaStep;
};

// Q(x,y) = power*(1+depth*sin(kx*x + ky*y - phase))
struct SIMFLUX_Modulation {
	float kx, ky;
	float depth;
	float phase;
	float power;
};

struct SIMFLUX_BlinkHMM_Params
{
	static DLL_EXPORT SIMFLUX_BlinkHMM_Params Compute(float T_on, float T_off, float fps);

	float prior; // p(on)
	float p_on2off, p_off2on;
	float p_on_threshold;
	float intensityThreshold;
	float excitationThreshold;
};

typedef Vector4f SIMFLUX_Theta;


template<typename TResult>
class ImageProcessingQueue;


struct SIMFLUX_EstimationResult
{
	typedef SIMFLUX_Theta Theta;
	enum { MaxSwitchFrames=2 };

	template<typename TLevMarResult>
	PLL_DEVHOST SIMFLUX_EstimationResult(Theta initialValue, TLevMarResult levMarResult, const Vector<float,16>& cov, const Vector<float,8>& crlb, int nsf) :
		estimate(levMarResult.estimate), initialValue(initialValue), crlb(crlb),
		iterations(levMarResult.iterations) {	}
	PLL_DEVHOST SIMFLUX_EstimationResult() { }

	Theta estimate;
	Theta initialValue;
	Vector<float, 4 + MaxSwitchFrames> crlb; // [X,Y,I,bg, SwitchFrameIntensities...]
	int iterations;
	uint64_t switchFrameMask, silmFrameMask;
	Int2 roiPosition;
};



#define SIMFLUX_MLE_CUDA 1				
#define SIMFLUX_MLE_FIXEDW 4

/*

imageData: float[imgw*imgw*epps*numspots]
backgroundMatrix: float[imgw*imgw*epps*numspots]
phi: float[epps*numspots]

*/
struct ImageQueueConfig;
class ISpotDetectorFactory;

//
// Computes expected values for each pixel and fisher matrix
// fiMatrix: float[16 * numspots]
// expectedValue: float[imgw*imgw*nframes*numspots]
CDLL_EXPORT void SIMFLUX_ASW_ComputeFisherMatrix(float * expectedValue, Gauss2D_FisherMatrix* fiMatrix, const SIMFLUX_Modulation* modulation,
	const SIMFLUX_Theta * theta, const Int3* roipos, int numspots, int nframes, const SIMFLUX_ASW_Params& p);


CDLL_EXPORT void SIMFLUX_ASW_ComputeMLE(const float * imageData, const SIMFLUX_Modulation*modulation,
	Gauss2D_EstimationResult *results, int numspots, int nframes, const SIMFLUX_ASW_Params& p,
	const SIMFLUX_Theta* initialTheta, const Int3* roipos, int flags,
	SIMFLUX_Theta* optimizerTrace=0, int traceBufferLength=0);


/* SILM pipeline in pieces...
- Gaussian fit
- Estimate intensities/bg per frame
- Estimate initial SILM theta
- HMM
- SILM fit (with specific frames)
*/

//CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb,
//	int numspots, const Vector2f* xy, float sigma, int imgw, int maxIterations, bool cuda)



CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth,
	float scale, int numProjAngles, const float *projectionAngles, float* output ,float* shifts);

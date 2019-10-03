#include "PSF.h"
#include "StringUtils.h"
#include "CudaUtils.h"
#include "SolveMatrix.h"

PSFBase::PSFBase(const std::vector<int>& sampleSize, int numConst, int thetaSize, int diagsize, const char * thetaFormat, Context* ctx) :
	thetaFormat(thetaFormat), sampleSize(sampleSize), sampleCount(1), numConstants(numConst), thetaSize(thetaSize), diagsize(diagsize), ContextObject(ctx)
{
	for (int s : sampleSize) sampleCount *= s;
}

PSFBase::~PSFBase()
{
}

int PSFBase::IndexInTheta(const char * name)
{
	const char *fmt = ThetaFormat();
	auto parts=StringSplit(fmt, ',');
	for (int i = 0; i < parts.size(); i++)
		if (parts[i] == name)
			return i;
	return -1;
}



CDLL_EXPORT void PSF_Delete(PSF * psf)
{
	delete psf;
}

CDLL_EXPORT const char * PSF_ThetaFormat(PSF * psf)
{
	return psf->ThetaFormat();
}

CDLL_EXPORT int PSF_ThetaSize(PSF * psf)
{
	return psf->ThetaSize();
}

CDLL_EXPORT int PSF_SampleCount(PSF * psf)
{
	return psf->SampleCount();
}

CDLL_EXPORT int PSF_SampleIndexDims(PSF* psf)
{
	return psf->SampleIndexDims();
}

CDLL_EXPORT int PSF_SampleSize(PSF* psf, int dim)
{
	return psf->SampleSize()[dim];
}

CDLL_EXPORT int PSF_NumConstants(PSF * psf)
{
	return psf->NumConstants();
}

CDLL_EXPORT int PSF_NumDiag(PSF* psf)
{
	return psf->DiagSize();
}

CDLL_EXPORT void PSF_ComputeExpectedValue(PSF * psf, int numspots, const float* theta, const float* constants, const int* spot_pos, float * ev)
{
	psf->ExpectedValue(ev, theta, constants, spot_pos, numspots);
}

CDLL_EXPORT void PSF_ComputeMLE(PSF * psf, int numspots, const float * sample, const float* constants, 
	const int* spot_pos, const float * initial, float * theta,  float *diagnostics, int* iterations, 
	float* trace, int traceBufLen)
{
	psf->Estimate(sample, constants, spot_pos, initial, theta, diagnostics, iterations, numspots, trace, traceBufLen);
}

CDLL_EXPORT void PSF_ComputeFisherMatrix(PSF * psf, int numspots, const float * theta, const float* constants, const int* spot_pos, float * fi)
{
	psf->FisherMatrix(theta, constants, spot_pos, numspots, fi);
}

CDLL_EXPORT void PSF_ComputeDerivatives(PSF * psf, int numspots, const float * theta, const float* constants, 
	const int* spot_pos, float * derivatives, float * ev)
{
	psf->Derivatives(derivatives, ev, theta, constants, spot_pos, numspots);
}

CDLL_EXPORT void PSF_DrawSpots(PSF * psf, const float * theta, const float* constants, const Int2* roipos, int numspots)
{
}


#pragma once

#include "PSF.h"

class NumericalDerivativesAdapter : public CUDA_PSF
{
	CUDA_PSF* psf;
	CUDA_PSF::VectorType<float>::type expectedValueBuffer;

public:
	NumericalDerivativesAdapter(CUDA_PSF* psf, std::vector<float> epsilon, int maxspots) : CUDA_PSF(maxspots), psf(psf) {}
	~NumericalDerivativesAdapter() { delete psf; }

};

CDLL_EXPORT CUDA_PSF* CUDA_PSF_NumericalDerivatives(CUDA_PSF* psf, const float* epsilon)
{
	return new NumericalDerivativesAdapter(psf, std::vector<float>(epsilon, epsilon + psf->ThetaSize()));
}

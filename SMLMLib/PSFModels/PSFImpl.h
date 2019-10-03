#pragma once

#include <streambuf>
#include "PSF.h"
#include "Estimation.h"
#include "ContainerUtils.h"
#include "ThreadUtils.h"


template<typename T, int K>
std::vector<int> makevector(Vector<T, K> v) {
	std::vector<int> r(K);
	for (int i = 0; i < K; i++)
		r[i] = v[i];
	return r;
}

template<typename TModel, typename TCalibration, typename TSampleOffset>
class CUDA_PSFImpl : public CUDA_PSF
{
public:
	typedef typename TCalibration Calibration;
	typedef TModel Model;
	typedef TSampleOffset SampleOffset;
	typedef typename Model::Theta Theta;
	typedef typename TModel::TSampleIndex SampleIndex;

	SampleOffset sampleOffset;
	SampleIndex roisize;
	float levmarInitialAlpha;

	CUDA_PSFImpl(const TSampleOffset& sampleOffset, SampleIndex roisize, float levMarInitialAlpha, int maxiterations) :
		CUDA_PSF( makevector(roisize),TModel::NumConstants, TModel::K, 0, maxiterations, TModel::ThetaFormat()),
		sampleOffset(sampleOffset), roisize(roisize), levmarInitialAlpha(levMarInitialAlpha)
	{}

	virtual const TCalibration& GetCalibration() = 0;
	
	void FisherMatrix(const float* d_theta, const float *d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		SampleIndex roisize=this->roisize;
		auto *fi = (typename FisherMatrixType<Theta>::type*)d_FI;
		auto roipos = (const SampleIndex*)d_roipos;
		int numconst = this->NumConstants();
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i*numconst]);
			fi[i] = ComputeFisherMatrix(model, roipos[i], ofs, theta[i]);
		}, 0, stream);
	}

	void ExpectedValue(float* d_image, const float* d_theta, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		auto roisize = this->roisize;
		int sc = SampleCount();
		int numconst = this->NumConstants();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i*numconst]);
			ComputeExpectedValue(theta[i], model, roipos[i], ofs, &d_image[sc*i] );
		}, 0, stream);
	}

	void Derivatives(float* d_deriv, float *d_expectedvalue, const float* d_theta, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		int smpcount = SampleCount();
		auto roisize = this->roisize;
		int numconst = this->NumConstants();
		int K = ThetaSize();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		LaunchKernel(numspots,  [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i*numconst]);
			ComputeDerivatives(theta[i], model, roipos[i], ofs, &d_deriv[i*smpcount*K], &d_expectedvalue[i*smpcount]);
		}, 0, stream);
	}

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, ThetaSize()], d_theta[numspots, ThetaSize()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		Theta* theta = (Theta*)d_theta;
		Theta* trace = (Theta*)d_trace;
		const Theta* initial = (const Theta*)d_initial;
		TCalibration calib = GetCalibration();
		int smpcount = SampleCount();
		auto roisize = this->roisize;
		SampleOffset ofs = sampleOffset;
		float levMarInitialAlpha = this->levmarInitialAlpha;
		int maxIterations = this->maxiterations;
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		int numconst = this->NumConstants();
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i*numconst]);
			auto r = ComputeMLE(&d_sample[i*smpcount], model, roipos[i], ofs, maxIterations, initial ? &initial[i] : 0, 
				&trace[traceBufLen*i], traceBufLen, levMarInitialAlpha);
			if(d_iterations) d_iterations[i] = r.iterations;
			theta[i] = r.estimate;
		}, 0, stream);
	}
/*	virtual void LimitTheta(const float* d_theta, const float* d_const, const int * d_roipos, int numspots, cudaStream_t stream) override
	{
		TCalibration calib = GetCalibration();
		Estimate* theta = (Estimate*)d_theta;
		int numconst = this->NumConstants();
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel mdl(roisize, calib, &d_const[i*numconst]);
			mdl.CheckLimits(theta);
		}, 0, stream);
}*/

};

template<typename TModel, typename TCalibration, typename TSampleOffset>
class PSFImpl : public PSF
{
public:
	typedef typename TCalibration Calibration;
	typedef typename TModel Model;
	typedef typename TSampleOffset SampleOffset;
	typedef typename TModel::TSampleIndex SampleIndex;

	SampleOffset sampleOffset;
	SampleIndex roisize;
	float levmarInitialAlpha;
	int maxiterations;
	typedef typename Model::Theta Theta;

	PSFImpl(const TSampleOffset& sampleOffset, SampleIndex roisize, float levMarInitialAlpha, int maxiterations) :
		PSF(makevector(roisize), TModel::NumConstants, TModel::K, 1, TModel::ThetaFormat()),
		sampleOffset(sampleOffset),roisize(roisize), levmarInitialAlpha(levMarInitialAlpha), maxiterations(maxiterations)
	{}

	virtual const TCalibration& GetCalibration() = 0;

	// Inherited via PSF
	virtual void FisherMatrix(const float *theta, const float *h_const, const int* spot_pos, int numspots, float * FI) override
	{
		const Theta* theta_ = (const Theta*)theta;
		const TCalibration& calib = GetCalibration();
		auto* fi = (typename FisherMatrixType<Theta>::type*)FI;
		auto roipos = (const typename TModel::TSampleIndex*)spot_pos;
		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*TModel::NumConstants]);
			fi[i] = ComputeFisherMatrix(model, roipos[i], sampleOffset, theta_[i]);
		});
	}

	virtual void ExpectedValue(float * expectedvalue, const float * theta, const float * h_const, const int* spot_pos, int numspots)  override
	{
		const Theta* theta_ = (const Theta*)theta;
		const TCalibration& calib = GetCalibration();
		auto roipos = (const typename TModel::TSampleIndex*)spot_pos;
		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*TModel::NumConstants]);
			ComputeExpectedValue(theta_[i], model, roipos[i], sampleOffset, &expectedvalue[model.SampleCount() *i]);
		});
	}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float *h_const, const int* spot_pos, int numspots) override
	{
		const Theta* theta_ = (const Theta*)theta;
		const TCalibration& calib = GetCalibration();
		auto roipos = (const typename TModel::TSampleIndex*)spot_pos;
		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*TModel::NumConstants]);
			ComputeDerivatives(theta_[i], model, roipos[i], sampleOffset, &deriv[model.SampleCount()*Theta::K*i], &expectedvalue[model.SampleCount()*i]);
		});
	}
	virtual void Estimate(const float * samples, const float *h_const, const int* spot_pos, const float * initial, float * theta,
		float* diag, int* iterations, int numspots, float * trace, int traceBufLen) override
	{
		Theta* trace_ = (Theta*)trace;
		Theta* theta_ = (Theta*)theta;
		const Theta* initial_ = (const Theta*)initial;
		const TCalibration& calib = GetCalibration();
		int smpcount = SampleCount();
		auto roipos = (const typename TModel::TSampleIndex*)spot_pos;

		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*TModel::NumConstants]);
			auto r = ComputeMLE(&samples[i*smpcount], model, roipos[i], sampleOffset, maxiterations, 
				initial_ ? &initial_[i] : (Theta*)0, &trace_[traceBufLen*i], traceBufLen, levmarInitialAlpha);
			if(iterations) iterations[i] = r.iterations;
			theta_[i] = r.estimate;
		});
	}
};


// PSF Implementation for models with a plain-old-data (POD) calibration type (no [cuda] memory management needed)
template<typename BasePSFImpl>
class SimpleCalibrationPSF : public BasePSFImpl
{
public:
	typedef typename BasePSFImpl::Calibration Calibration;
	typedef typename BasePSFImpl::SampleIndex SampleIndex;
	typedef typename BasePSFImpl::SampleOffset SampleOffset;
	Calibration calib;

	const Calibration& GetCalibration() override { return calib; }

	SimpleCalibrationPSF(const Calibration& calib, const SampleOffset& smpofs, SampleIndex roisize, int maxIterations, float levmarInitialAlpha) :
		BasePSFImpl(smpofs, roisize, levmarInitialAlpha, maxIterations), calib(calib)
	{}
};

// Wraps a CUDA_PSF into a host-memory PSF
class CUDA_PSF_Wrapper : public PSF {
	CUDA_PSF* psf;
public:
	CUDA_PSF_Wrapper(CUDA_PSF* psf);
	~CUDA_PSF_Wrapper();

	CUDA_PSF* GetCUDA_PSF() override { return psf; }

	// Inherited via PSF
	virtual void FisherMatrix(const float * h_theta, const float* h_const, const int* spot_pos, int numspots, float * h_FI) override;
	virtual void ExpectedValue(float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots) override;
	virtual void Derivatives(float * h_deriv, float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots) override;
	virtual void Estimate(const float * h_sample, const float* h_const, const int* spot_pos, const float * h_initial, 
		float * h_theta, float * h_diag, int* iterations, int numspots, float* d_trace, int traceBufLen) override;
};



class CenterOfMassEstimator : public CUDA_PSF {
public:
	CenterOfMassEstimator(int roisize);

	// Implement COM
	virtual void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream) override;

	// All these do nothing
	virtual void FisherMatrix(const float* d_theta, const float* d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream) {}
	virtual void ExpectedValue(float * expectedvalue, const float * d_theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
protected:
	int roisize;
};

// pSMLM
class PhasorEstimator : public CUDA_PSF {
public:
	PhasorEstimator(int roisize);

	virtual void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream) override;

	// All these do nothing
	virtual void FisherMatrix(const float* d_theta, const float* d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream) {}
	virtual void ExpectedValue(float * expectedvalue, const float * d_theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
protected:
	int roisize;
};



// Wraps another CUDA_PSF and copies all samples to the diagnostics array
class CopyROI_PSF : public CUDA_PSF {
public:
	CopyROI_PSF(CUDA_PSF* org);

	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override;

	// Uses the provided model
	virtual void FisherMatrix(const float* d_theta, const float* d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream)
	{ psf->FisherMatrix(d_theta, d_const, d_roipos, numspots, d_FI, stream); }
	virtual void ExpectedValue(float * expectedvalue, const float * d_theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{ psf->ExpectedValue(expectedvalue, d_theta, d_const, d_roipos, numspots, stream); }
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{ psf->Derivatives(deriv, expectedvalue, theta, d_const, d_roipos, numspots, stream);  }
protected:
	CUDA_PSF* psf;
};

CDLL_EXPORT PSF* CreatePhasorEstimator(int roisize, Context* ctx);
CDLL_EXPORT PSF* CreateCenterOfMassEstimator(int roisize, Context* ctx);
CDLL_EXPORT PSF * CopyROI_CreatePSF(PSF* model, Context* ctx);




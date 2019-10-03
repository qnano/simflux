#include "PSFModels/PSF.h"
#include "ThreadUtils.h"
#include "Estimation.h"
#include "PSFModels/Gaussian/GaussianPSFModels.h"
#include "CameraCalibration.h"

// A model that only fits background. 
struct BgModel : public Gauss2D_PSFModel<float, 1>
{
	PLL_DEVHOST T StopLimit(int k) const
	{
		return 1e-4f;
	}

	PLL_DEVHOST BgModel(Int2 roisize) : Gauss2D_PSFModel(roisize) {}

	PLL_DEVHOST void CheckLimits(Theta& t) const
	{
		if (t.elem[0] < 1e-8f) t.elem[0] = 1e-8f;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Theta theta, const TSampleIndex& roipos) const
	{
		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				const T firstOrder[] = { 1 };
				cb(Int2{ y,x }, theta[0], firstOrder);
			}
		}
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeSecondDerivatives(TCallback cb, Theta theta, const TSampleIndex& roipos) const
	{
		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				// mu=bg
				// dmu/dbg = 1
				// 
				const T firstOrder[] = { 1 };
				const T secondOrder[] = { 0 };
				cb(Int2{ y,x }, theta[0], firstOrder, secondOrder);
			}
		}
	}
};

// Computes likelihoods of model and background-only, and stores them in the diagnostics output array
template<typename TSmpOfs>
class GLRT_PSF : public CUDA_PSF {
public:
	GLRT_PSF(CUDA_PSF* org, TSmpOfs smpofs) : 
		CUDA_PSF(org->SampleSize(), org->NumConstants(), org->ThetaSize(), 3, org->MaxIterations(), org->ThetaFormat()),
		psf(org), smpofs(smpofs)
	{

	}

	struct Buffers {
		Buffers(int psfsmpcount, int numspots) :
			expval(psfsmpcount*numspots),
			numspots(numspots) {}
		DeviceArray<float> expval;
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, Buffers> streamData;
	TSmpOfs smpofs;

	Buffers* GetBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}

			if (it == streamData.end())
				it = streamData.emplace(stream, Buffers(SampleCount(), numspots)).first;
			return &it->second;
		});
	}

	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, 
		float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		psf->Estimate(d_sample, d_const, d_roipos, d_initial, d_theta, d_diag, d_iterations, numspots, d_trace, traceBufLen, stream);
		Buffers* b = GetBuffers(stream, numspots);
		Vector3f* llbg = (Vector3f*)d_diag;

		psf->ExpectedValue(b->expval.data(), d_theta, d_const, d_roipos, numspots, stream);
		const float* expval = b->expval.data();

		int roisizeX = SampleSize()[1], roisizeY = SampleSize()[0];
		const Int2* roipos = (const Int2*)d_roipos;
		TSmpOfs smpofs = this->smpofs;
		int smpcount = SampleCount();
		LaunchKernel(numspots, [=]__device__(int i) {
			const float* ev = &expval[smpcount*i];
			const float* smp = &d_sample[i*smpcount];
			BgModel model({ roisizeY,roisizeX });
//			auto r = NewtonRaphson(smp, { 1.0f }, model, roipos[i], smpofs, 10);
			auto r = LevMarOptimize(smp, { 1.0f }, model, roipos[i], smpofs, 20);
			llbg[i][1] = ComputeLogLikelihood(r.estimate, smp, model, roipos[i], smpofs);
			llbg[i][2] = r.estimate[0];

			float ll_on = 0.0f;
			for (int y = 0; y < roisizeY; y++)
			{
				for (int x = 0; x < roisizeX; x++) {
					float mu = ev[y*roisizeX + x];
					float readnoise = smpofs.Get({ y,x }, roipos[i]);
					mu += readnoise;
					if (mu < 0.0f) mu = 0.0f;
					float d = smp[y*roisizeX + x] + readnoise;
					ll_on += d * log(mu) - mu;
				}
			}
			llbg[i][0] = ll_on;
		}, 0, stream);
	}

	// Uses the provided model
	virtual void FisherMatrix(const float* d_theta, const float* d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream)
	{
		psf->FisherMatrix(d_theta, d_const, d_roipos, numspots, d_FI, stream);
	}
	virtual void ExpectedValue(float * expectedvalue, const float * d_theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{
		psf->ExpectedValue(expectedvalue, d_theta, d_const, d_roipos, numspots, stream);
	}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{
		psf->Derivatives(deriv, expectedvalue, theta, d_const, d_roipos, numspots, stream);
	}
protected:
	CUDA_PSF* psf;
};







CDLL_EXPORT PSF * GLRT_CreatePSF(PSF* model, Context* ctx, sCMOS_Calibration* calib)
{
	CUDA_PSF* psf;
	if (calib) {
		auto smpofs = calib->GetSampleOffset();
		psf = new GLRT_PSF<decltype(smpofs)>(model->GetCUDA_PSF(), smpofs);
	} else
		psf = new GLRT_PSF<SampleOffset_None<float>>(model->GetCUDA_PSF(), SampleOffset_None<float>());
	return PSF_WrapCUDA_PSF(psf);
}


#include "simflux/SIMFLUX.h"
#include "simflux/ExcitationModel.h"
#include "PSFModels/PSF.h"
#include "CudaUtils.h"
#include "PSFModels/PSFImpl.h"
#include "SIMFLUX_PSF.h"
#include "simflux/SIMFLUX_Models.h"
#include "PSFModels/Gaussian/GaussianPSFModels.h"

static std::vector<int> samplesize(std::vector<int> org, int numPatterns)
{
	org.insert(org.begin(), numPatterns);
	return org;
}



template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST bool LevMarStep(int K, T* theta, const T* sample, int iteration, const TModel& model, const typename TModel::TSampleIndex& roipos,
	float lambdaStep, TSampleOffset sampleOffset, T* workspace, const T* minTheta, const T* maxTheta) {

	T* alpha = workspace;
	T* LU = alpha + (K*K);
	T* beta = LU + (K*K);
	T* thetaStep = beta + K;

	ComputeLevMarAlphaBeta(alpha, beta, theta, sample, model, roipos, sampleOffset);

	for (int k = 0; k < K; k++)
		alpha[k*K + k] *= 1 + lambdaStep;

	if (!Cholesky(K, alpha, LU))
		return false;
	if (!SolveCholesky(LU, beta, thetaStep))
		return false;

	bool smallDeltas = true;
	for (int k = 0; k < K; k++) {
		if (abs(thetaStep[k]) > model.StopLimit(k)) {
			smallDeltas = false;
			break;
		}
	}

	for (int k = 0; k < K; k++)
		theta[k] += thetaStep[k];
	if (iteration > 0 && smallDeltas)
		return false;

	return theta;
}

class StoredDerivativesModel {
public:

	StoredDerivativesModel(int K, int smpcount, const float* deriv, const float* expVal)
		:K(K),smpcount(smpcount)
	{}
	PLL_DEVHOST int SampleIndex(int i) const { return i; }
	
	typedef const float* Theta;
	int K, smpcount;
	const float* deriv, *expVal;

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Theta theta, const int& roipos) const
	{
		for (int i = 0; i < smpcount; i++)
		{
//			float mu = theta[0] * psf[i] + theta[1];
	//		const float firstOrder[] = { psf[i], 1 };
		//	cb(i, mu, firstOrder);
		}
	}

};

class SIMFLUX_CUDA_PSF : public CUDA_PSF
{
public:
	Int4 xyIBg;
	std::vector<SIMFLUX_Modulation> mod;
	DeviceArray<SIMFLUX_Modulation> d_mod;
	CUDA_PSF* psf;
	bool simfluxFit;

	struct DeviceBuffers {
		DeviceBuffers(int baseSmpCount, int baseSmpDims, int numspots, int baseK, int numPatterns) :
			summed(baseSmpCount*numspots),
			expectedValue(baseSmpCount*numspots),
			readnoise(baseSmpCount*numspots),
			tmpTheta(baseK*numspots),
			excitations(numPatterns*numspots),
			baseroipos(baseSmpDims*numspots),
			derivatives(baseK*baseSmpCount*numspots),
			numspots(numspots) {}
		DeviceArray<float> summed, expectedValue, tmpTheta, readnoise;
		DeviceArray<int> baseroipos;
		DeviceArray<float> derivatives;
		DeviceArray<Vector3f> excitations;
		int numspots;
	};
	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;

	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots)
	{
		if (maxspots < numspots) {
			SetMaxSpots(numspots);
			if (streamData.find(stream) != streamData.end())
				streamData.erase(stream);
		}

		auto it = streamData.find(stream);

		if (it == streamData.end())
			it = streamData.emplace(stream, DeviceBuffers(psf->SampleCount(), psf->SampleIndexDims(), maxspots, psf->ThetaSize(), (int) mod.size())).first;

		return &it->second;
	}

	SIMFLUX_CUDA_PSF(CUDA_PSF* psf, std::vector<SIMFLUX_Modulation> mod, Int4 xyIBg, bool simfluxFit) :
		CUDA_PSF(samplesize(psf->SampleSize(), (int)mod.size()),
			psf->NumConstants(), psf->ThetaSize(), 2 * (int)mod.size(), psf->MaxIterations(), psf->ThetaFormat()),
			xyIBg(xyIBg), mod(mod), psf(psf), d_mod(mod), simfluxFit(simfluxFit)
	{
		assert(xyIBg[3] >= 0);
		assert(psf->SampleIndexDims() == 2);
	}

	void SetMaxSpots(int maxspots)
	{
		this->maxspots = maxspots;
		streamData.clear();
		CUDA_PSF::SetMaxSpots(maxspots);
	}

	void ComputeThetaAndExcitation(cudaStream_t stream, DeviceBuffers& db, const float* d_theta, const float* d_const, const int* d_roipos, int numspots)
	{
		int K = ThetaSize();
		// evaluate the actual PSF
		float* d_psf_ev = db.expectedValue.data();
		float* d_psf_theta = db.tmpTheta.data();
		Int2* d_psf_roipos = (Int2*)db.baseroipos.data();
		auto xyIBg_ = xyIBg;
		int numPatterns = (int)mod.size();
		const auto d_mod_ = d_mod.data();
		Vector3f* d_exc = db.excitations.data();
		
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int k = 0; k < K; k++)
				d_psf_theta[i*K + k] = d_theta[i*K + k];
			d_psf_theta[i*K + xyIBg_[3]] = 0;
			d_psf_theta[i*K + xyIBg_[2]] = 1.0f;
			d_psf_roipos[i] = { d_roipos[i * 3 + 1],d_roipos[i * 3 + 2] };
		}, 0, stream);
		
		DeviceArray<float2> xypos(numspots);
		float2* xypos_ = xypos.data();

		LaunchKernel(numspots, numPatterns, [=]__device__(int i, int p) {
			const int * roipos = &d_roipos[i * 3];
			FixedFunctionExcitationPatternModel epModel(numPatterns, d_mod_);
			int e = roipos[0] + p;
			if (e > numPatterns) e -= numPatterns; // cheap % (% is expensive on cuda)
			float q, dqdx, dqdy;
			float2 xy = make_float2( d_theta[i*K + xyIBg_[0]] + roipos[2], d_theta[i*K + xyIBg_[1]] + roipos[1] );
			xypos_[i] = xy;
//			printf("x=%f, y=%f\n", xy.x, xy.y);
			epModel.ExcitationPattern(q, dqdx, dqdy, e, xy);
			d_exc[i*numPatterns + p] = { q,dqdx,dqdy };
		}, 0, stream);

		cudaStreamSynchronize(stream);
		auto chk = db.excitations.ToVector();
		for (int j = 0; j < chk.size(); j++)
		{
			DebugPrintf("%d: ", j);  PrintVector(chk[j]);
		}
	}

	virtual void ExpectedValue(float * expectedvalue, const float * d_theta, const float * d_const, 
		const int * d_roipos, int numspots, cudaStream_t stream) override
	{
		if (numspots > maxspots)
			SetMaxSpots(numspots);

		auto& db = *GetDeviceBuffers(stream,numspots);
		ComputeThetaAndExcitation(stream, db, d_theta, d_const, d_roipos, numspots);
		float* d_psf_theta = db.tmpTheta.data();
		float* d_psf_ev = db.expectedValue.data();
		Int2* d_psf_spotpos = (Int2*)db.baseroipos.data();
		psf->ExpectedValue(d_psf_ev, d_psf_theta, d_const, (int*)d_psf_spotpos, numspots, stream);

		// compute excitation values and generate the SIM-ed expected values
		int numPatterns = (int)mod.size();
		int smpcount = psf->SampleCount();
		auto xyIBg_ = xyIBg;
		int K = ThetaSize();
		Vector3f* d_exc = db.excitations.data();
		float bg = 1.0f / numPatterns;

		LaunchKernel(numspots, numPatterns, smpcount, [=]__device__(int i, int p, int smp) {
			Vector3f exc = d_exc[i*numPatterns+p];
			expectedvalue[smpcount*numPatterns*i + smpcount * p + smp] =
				d_psf_ev[smpcount*i + smp] * exc[0] * d_theta[i*K+xyIBg_[2]] + d_theta[i*K + xyIBg_[3]] * bg;
		}, 0, stream);
	}

	virtual void Derivatives(float * d_deriv, float * d_expectedvalue, const float * d_theta, 
		const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) override
	{
		if (numspots > maxspots)
			SetMaxSpots(numspots);

		auto& sd = *GetDeviceBuffers(stream, numspots);
		ComputeThetaAndExcitation(stream, sd, d_theta, d_const, d_roipos, numspots);

		float* d_psf_theta = sd.tmpTheta.data();
		float* d_psf_ev = sd.expectedValue.data();
		float* d_psf_deriv = sd.derivatives.data();
		Int2* d_psf_spotpos = (Int2*)sd.baseroipos.data();
		// Compute base psf derivatives with I=1, bg=0
		psf->Derivatives(d_psf_deriv, d_psf_ev, d_psf_theta, d_const, (int*)d_psf_spotpos, numspots, stream);

		int numPatterns = (int)mod.size();
		int psfSmpCount = psf->SampleCount();
		auto xyIBg_ = xyIBg;
		int K = ThetaSize();
		Vector3f* d_exc = sd.excitations.data();
		float bg = 1.0f / numPatterns;
		int sfSmpCount = numPatterns * psfSmpCount;

		LaunchKernel(numspots, numPatterns, psfSmpCount, [=]__device__(int i, int p, int smp) {
			float* spot_deriv = &d_deriv[sfSmpCount*K *i];
			float* spot_ev = &d_expectedvalue[sfSmpCount*i];
			const float* spot_psf_deriv = &d_psf_deriv[psfSmpCount*K*i];
			const float* spot_psf_ev = &d_psf_ev[psfSmpCount*i];
			Vector3f exc = d_exc[i*numPatterns + p];

			float spotIntensity = d_theta[i*K + xyIBg_[2]];
			float spotBg = d_theta[i*K + xyIBg_[3]];
			float psf_ev = d_psf_ev[psfSmpCount*i + smp]  * exc[0];
			spot_ev[psfSmpCount*p + smp] = psf_ev * spotIntensity + spotBg * bg;

			for (int k = 0; k < K; k++) {
				float q_deriv = 0.0f;
				if (xyIBg_[0] == k) q_deriv = exc[1];
				else if (xyIBg_[1] == k) q_deriv = exc[2];
				spot_deriv[sfSmpCount*k + psfSmpCount * p + smp] =
					spotIntensity * (spot_psf_deriv[psfSmpCount * k + smp] * exc[0] + spot_psf_ev[smp] * psf_ev * q_deriv);
			} 
		}, 0, stream);
	}


	void Estimate(const float * d_sample, const float * d_const, const int* d_roipos, const float * d_initial,
		float * d_theta, float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
	{
		DeviceBuffers* db = GetDeviceBuffers(stream, numspots);

		// Sum samples
		int orgSmpCount = psf->SampleCount();
		int numPatterns = (int)this->mod.size();
		float* d_sum = db->summed.data();
		LaunchKernel(numspots, orgSmpCount, [=]__device__(int i, int j) {
			const float* smp = &d_sample[orgSmpCount*numPatterns*i];
			float sum = 0.0f;
			for (int k = 0; k < numPatterns; k++)
				sum += smp[orgSmpCount*k + j];
			d_sum[orgSmpCount*i + j] = sum;
		}, 0, stream);

		// Compute roipos[:,1:]
		int* baseroipos = db->baseroipos.data();
		int nsmpdims = psf->SampleIndexDims();
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int j = 0; j < nsmpdims; j++)
				baseroipos[i*nsmpdims + j] = d_roipos[(1 + nsmpdims)*i + j + 1];
		}, 0, stream);

		psf->Estimate(d_sum, d_const, db->baseroipos.data(), d_initial, d_theta, 0, d_iterations, numspots, d_trace, traceBufLen, stream);

		// Compute theta with I=1 and bg=0
		int intensityIndex = this->xyIBg[2];
		int bgIndex = this->xyIBg[3];
		int K = this->ThetaSize();
		float* adjTheta = db->tmpTheta.data();
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int j = 0; j < K; j++)
				adjTheta[i*K + j] = d_theta[i*K + j];
			adjTheta[i*K + intensityIndex] = 1.0f;
			adjTheta[i*K + bgIndex] = 0.0f;
		}, 0, stream);

		// Compute expected values with I=1,bg=0.
		float* computed_psf = db->expectedValue.data();
		psf->ExpectedValue(computed_psf, adjTheta, d_const, baseroipos, numspots, stream);
		//Compute readnoise
		LaunchKernel(numspots, [=]__device__(int i) {
			adjTheta[i*K + intensityIndex] = 0.0f;
		}, 0, stream);
		psf->ExpectedValue(db->readnoise.data(), adjTheta, d_const, baseroipos, numspots, stream);
		float *readnoise = db->readnoise.data();
		LaunchKernel(numspots*orgSmpCount, [=]__device__(int i) {
			computed_psf[i] -= readnoise[i];
		}, 0, stream);

		// Estimate intensities and backgrounds 
		Vector2f* IBg = (Vector2f*)d_diag;
		LaunchKernel(numspots, numPatterns, [=]__device__(int spot, int pat) {
			IntensityBgModel model({ orgSmpCount,1 }, &computed_psf[orgSmpCount*spot]);
			const float* smp = &d_sample[orgSmpCount*numPatterns*spot + orgSmpCount * pat];
			Vector2f initial{ 1.0f, d_theta[spot*K + bgIndex] / numPatterns };
			auto smpofs = SampleOffset_None<float>();// SampleOffset_sCMOS_ROI<float>{ &readnoise[numspots*orgSmpCount] };
			auto r = LevMarOptimize(smp, initial, model, { 0,0 }, smpofs, 15);
			IBg[spot*numPatterns + pat] = r.estimate;
		}, 0, stream);

		if (simfluxFit)
		{
			// Theta already contains summed-psf estimate at this point
			//Derivatives(d_deriv, d_ev, d_theta, d_const, d_roipos, numspots, stream);

			LaunchKernel(numspots, [=]__device__(int i) {
				
			}, 0, stream);

		}
	}
};


CDLL_EXPORT PSF* SIMFLUX2D_PSF_Create(PSF* original, SIMFLUX_Modulation* mod, int num_patterns, 
	const int * xyIBg_indices, bool simfluxFit, Context* ctx)
{
	CUDA_PSF* original_cuda = original->GetCUDA_PSF();
	if (!original_cuda) return 0;

	auto mod_ = std::vector<SIMFLUX_Modulation>(mod, mod + num_patterns);
	Int4 xyIBg_(xyIBg_indices[0], xyIBg_indices[1], xyIBg_indices[2], xyIBg_indices[3]);

	PSF* psf = new CUDA_PSF_Wrapper(
		new SIMFLUX_CUDA_PSF(original_cuda, mod_, xyIBg_, simfluxFit));

	if (ctx) psf->SetContext(ctx);
	return psf;
}


std::vector<int> prependInt(std::vector<int> v, int a) {
	v.insert(v.begin(), a);
	return v;
}


template<typename BaseOffset>
class SampleOffset_Multiply
{
public:
	BaseOffset base;
	float factor;
	PLL_DEVHOST SampleOffset_Multiply(BaseOffset b, float factor) :base(b), factor(factor) {}

	PLL_DEVHOST float Get(Int2 samplepos, Int2 roipos) const {
		return base.Get(samplepos, roipos)*factor;
	}
	PLL_DEVHOST float Get(Int3 samplepos, Int3 roipos) const {
		return base.Get(samplepos, roipos)*factor;
	}
};

template<typename SampleOffset>
class SIMFLUX_Gauss2D_CUDA_PSF : public CUDA_PSF 
{
public:
	typedef Gauss2D_Theta Theta;
	typedef Int3 SampleIndex;
	typedef SIMFLUX_Calibration TCalibration;
	typedef SIMFLUX_Model TModel;

	bool simfluxFit; // If false, only per-pattern intensities/backgrounds are estimated and a Gaussian fit is done on the summed frames
	bool defineStartEnd=false;
	float2 sigma;
	DeviceArray<SIMFLUX_Modulation> d_mod;
	SampleOffset sampleOffset;
	int numframes, roisize;
	float levmarInitialAlpha=1.0f;

	struct Buffers {
		Buffers(int psfsmpcount, int numspots) :
			summed(psfsmpcount*numspots),
			numspots(numspots) {}
		DeviceArray<float> summed;
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, Buffers> streamData;

	Buffers* GetBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}			

			if (it == streamData.end())
				it = streamData.emplace(stream, Buffers(roisize*roisize, numspots)).first;
			return &it->second;
		});
	}

	SIMFLUX_Gauss2D_CUDA_PSF(int roisize, int numframes, std::vector<SIMFLUX_Modulation> mod, SampleOffset smpofs,
							float2 sigma, bool simfluxFit, bool defineStartEnd, int maxIterations) :
		CUDA_PSF({ numframes,roisize,roisize }, defineStartEnd ? 2 : 0, 4, numframes * 4, maxIterations, Gauss2D_Model_XYIBg::ThetaFormat()),
		d_mod(mod), 
		sampleOffset(smpofs),
		roisize(roisize), 
		numframes(numframes), 
		simfluxFit(simfluxFit), 
		sigma(sigma), 
		defineStartEnd(defineStartEnd)
	{}

	SIMFLUX_Calibration GetCalibration() 
	{
		return { FixedFunctionExcitationPatternModel((int)d_mod.size(), d_mod.data()), sigma };
	}

	void FisherMatrix(const float* d_theta, const float *d_const, const int* d_roipos, int numspots, float* d_FI, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		int roisize = this->roisize;
		int numframes = this->numframes;
		auto *fi = (typename FisherMatrixType<Theta>::type*)d_FI;
		auto roipos = (const SampleIndex*)d_roipos;
		int numconst = this->NumConstants();
		if (simfluxFit) {
			LaunchKernel(numspots, [=]__device__(int i) {
				TModel model(roisize, calib, 0, numframes, numframes);
				fi[i] = ComputeFisherMatrix(model, roipos[i], ofs, theta[i]);
			}, 0, stream);
		}
		else {
			LaunchKernel(numspots, [=]__device__(int i) {
				Gauss2D_Model_XYIBg model({ roisize,roisize }, calib.sigma);
				fi[i] = ComputeFisherMatrix(model, { roipos[i][1],roipos[i][2] }, ofs, theta[i]);
			}, 0, stream);
		}
	}

	void ExpectedValue(float* d_image, const float* d_theta, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		int roisize = this->roisize;
		int numframes = this->numframes;
		int sc = SampleCount();
		int numconst = this->NumConstants();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		if (simfluxFit) {
			LaunchKernel(numspots, [=]__device__(int i) {
				TModel model(roisize, calib, 0, numframes, numframes);
				ComputeExpectedValue(theta[i], model, roipos[i], ofs, &d_image[sc*i]);
			}, 0, stream);
		}
		else {
			assert(0);
		}
	}

	void Derivatives(float* d_deriv, float *d_expectedvalue, const float* d_theta, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Theta* theta = (const Theta*)d_theta;
		TCalibration calib = GetCalibration();
		SampleOffset ofs = sampleOffset;
		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		int numconst = this->NumConstants();
		int K = ThetaSize();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		if (simfluxFit) {
			LaunchKernel(numspots, [=]__device__(int i) {
				TModel model(roisize, calib, 0, numframes, numframes);
				ComputeDerivatives(theta[i], model, roipos[i], ofs, &d_deriv[i*smpcount*K], &d_expectedvalue[i*smpcount]);
			}, 0, stream);
		}
		else {
			assert(0);
		}
	}

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, ThetaSize()], d_theta[numspots, ThetaSize()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag,
		int *iterations,int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		Buffers* db = GetBuffers(stream, numspots);

		Theta* theta = (Theta*)d_theta;
		Theta* trace = (Theta*)d_trace;
		TCalibration calib = GetCalibration();
		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		SampleOffset ofs = sampleOffset;
		float levMarInitialAlpha = this->levmarInitialAlpha;
		int maxIterations = this->maxiterations;
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
	
		float* d_sums = db->summed.data();

		LaunchKernel(numspots, PALALA(int i) {
			Gauss2D_Model_XYIBg model({ roisize,roisize }, calib.sigma);

			const float* smp = &d_sample[i*smpcount];

			float* smp_sum = &d_sums[i*roisize*roisize];
			ComputeImageSums(smp, smp_sum, roisize, roisize, numframes);
			Int2 roiposYX{ roipos[i][1], roipos[i][2] };

			Theta estim;
			if (d_initial)
			{
				estim = ((Theta*)d_initial)[i];
				iterations[i] = 0;
			}
			else{
				// Don't use the sCMOS offsets for summed frames
				SampleOffset_Multiply<SampleOffset> summedSmpOfs(ofs, 1 / sqrtf((float)numframes));
				auto com = model.ComputeInitialEstimate(smp_sum, 0, roiposYX, summedSmpOfs);
//				auto com = ComputePhasorEstim(smp_sum, roisize, roisize);
				auto initialValue = Vector4f{ com[0],com[1],com[2] * 0.9f, 0.0f };

				auto r = LevMarOptimize(smp_sum, initialValue, model, roiposYX, summedSmpOfs, maxIterations,
					&trace[traceBufLen*i], traceBufLen, levMarInitialAlpha);

				estim = r.estimate;
				iterations[i] = r.iterations;
			}

			float* psf = smp_sum; // don't need the summed frames anymore at this point.
			IntensityBgModel::ComputeGaussianPSF(calib.sigma, estim[0], estim[1], roisize, psf);
			theta[i] = estim;

			float* spot_diag = &d_diag[i*numframes * 4];
			for (int j = 0; j < numframes; j++) {
				IntensityBgModel ibg_model({ roisize,roisize }, psf);
				auto ibg_r = LevMarOptimize(&smp[roisize*roisize*j], { 1.0f, 0.0f }, ibg_model, roiposYX, ofs, maxIterations);
				auto ibg_crlb = ComputeCRLB(ComputeFisherMatrix(ibg_model, roiposYX, ofs, ibg_r.estimate));
				spot_diag[j * 4 + 0] = ibg_r.estimate[0];
				spot_diag[j * 4 + 1] = ibg_r.estimate[1];
				spot_diag[j * 4 + 2] = ibg_crlb[0];
				spot_diag[j * 4 + 3] = ibg_crlb[1];
			}

		},0,stream);

		if (simfluxFit)
		{
			if (defineStartEnd) {
				Vector2f* const_ = (Vector2f*)d_const;
				LaunchKernel(numspots, [=]__device__(int i) {
					int start = (int)const_[i][0],  end = (int)const_[i][1];
					SIMFLUX_Model model(roisize, calib, start, end, numframes);
					const float* smp = &d_sample[i*smpcount];

					auto r = LevMarOptimize(smp, theta[i], model, roipos[i], ofs, maxIterations,
						&trace[traceBufLen*i], traceBufLen, levMarInitialAlpha);
					theta[i] = r.estimate;
					iterations[i] = r.iterations;
				}, 0, stream);
			}
			else {
				LaunchKernel(numspots, [=]__device__(int i) {
					TModel model(roisize, calib, 0, numframes, numframes);
					const float* smp = &d_sample[i*smpcount];

					auto r = LevMarOptimize(smp, theta[i], model, roipos[i], ofs, maxIterations,
						&trace[traceBufLen*i], traceBufLen, levMarInitialAlpha);
					theta[i] = r.estimate;
					iterations[i] = r.iterations;
				}, 0, stream);
			}
		}
	}
};

CDLL_EXPORT PSF* SIMFLUX2D_Gauss2D_PSF_Create(SIMFLUX_Modulation* mod, int num_patterns, float sigmaX ,float sigmaY, 
	int roisize, int numframes, bool simfluxFit, bool defineStartEnd,  sCMOS_Calibration* scmos_calib, Context* ctx)
{
	int maxIterations = 100;
	CUDA_PSF *cpsf;

	std::vector<SIMFLUX_Modulation> modulation(mod, mod + num_patterns);
	float2 sigma { sigmaX,sigmaY };

	if (scmos_calib) {
		cpsf = new SIMFLUX_Gauss2D_CUDA_PSF<SampleOffset_sCMOS<float>>(roisize, numframes,
			modulation, scmos_calib->GetSampleOffset(), sigma, simfluxFit, defineStartEnd, maxIterations);
	}
	else {
		cpsf = new SIMFLUX_Gauss2D_CUDA_PSF<SampleOffset_None<float>>(roisize, numframes,
			modulation, SampleOffset_None<float>(), sigma, simfluxFit, defineStartEnd, maxIterations);
	}

	PSF* psf = new CUDA_PSF_Wrapper(cpsf);
	if (ctx) psf->SetContext(ctx);
	return psf;
}



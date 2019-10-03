#include "PSFImpl.h"
#include "CudaUtils.h"
#include <unordered_map>

// For ComputeCOM
#include "Gaussian/GaussianPSFModels.h"

CDLL_EXPORT PSF* PSF_WrapCUDA_PSF(CUDA_PSF* psf)
{
	return new CUDA_PSF_Wrapper(psf);
}



CUDA_PSF::DeviceBuffers::DeviceBuffers(int smpcount, int numspots, int thetasize)
	: derivatives(smpcount*thetasize*numspots), expectedValue(smpcount*numspots)
{}

CUDA_PSF::DeviceBuffers::~DeviceBuffers() {}


CUDA_PSF::DeviceBuffers* CUDA_PSF::GetDeviceBuffers(cudaStream_t stream, int numspots)
{
	if (maxspots < numspots) {
		SetMaxSpots(numspots);
		if (streamData.find(stream) != streamData.end())
			streamData.erase(stream);
	}

	auto it = streamData.find(stream);

	if (it == streamData.end())
		it = streamData.emplace(stream, DeviceBuffers(SampleCount(), maxspots, ThetaSize())).first;

	return &it->second;
}


void CUDA_PSF::SetMaxSpots(int maxspots)
{
	this->maxspots = maxspots;
}

CUDA_PSF::CUDA_PSF(const std::vector<int>& sampleSize, int numConst, int thetaSize, int diagsize, int maxiterations,
	const char* thetaFormat)
	: PSFBase(sampleSize, numConst, thetaSize, diagsize, thetaFormat), 
	maxspots(0), maxiterations(maxiterations)
{}

CUDA_PSF::~CUDA_PSF()
{}

void CUDA_PSF::FisherMatrix(const float * d_theta, const float * d_const, const int * d_roipos, int numspots, float * d_FI, cudaStream_t stream)
{
	auto buffers = GetDeviceBuffers(stream, numspots);

	float* d_deriv = buffers->derivatives.data();
	float* d_ev = buffers->expectedValue.data();

	Derivatives(d_deriv, d_ev, d_theta, d_const, d_roipos, numspots, stream);

	int K = ThetaSize();
	int smpcount = SampleCount();

	LaunchKernel(numspots, [=]__device__(int spot) {
		float* fi = &d_FI[K*K*spot];
		for (int i = 0; i < K*K; i++)
			fi[i] = 0;

		const float *spot_deriv = &d_deriv[spot*smpcount*K];

		for (int i = 0; i < smpcount; i++) {
			float mu = d_ev[spot * smpcount + i];
			auto jacobian = [=](int j) { return spot_deriv[smpcount*j + i]; };

			float inv_mu_c = 1.0f / (mu > 1e-8f ? mu : 1e-8f);
			for (int i = 0; i < K; i++) {
				for (int j = i; j < K; j++) {
					const float fi_ij = jacobian(i) * jacobian(j) * inv_mu_c;
					fi[K*i + j] += fi_ij;
				}
			};
		}
		// fill below diagonal
		for (int i = 1; i < K; i++)
			for (int j = 0; j < i; j++)
				fi[K*i + j] = fi[K*j + i];
	}, 0, stream);
}

PLL_DEVHOST int LevMarStepWorkspaceSize(int K)
{
	return K * K * 2 + K * 2;
}


DLL_EXPORT void CUDA_PSF::FisherToCRLB(const float *d_fi, float* d_crlb, int numspots, cudaStream_t stream)
{
	int numThreads = 16;
	int K = ThetaSize();
	int sharedMemPerSpot = 2 * K * K + K + 1;
	LaunchKernel(numspots, [=]__device__(int i) {
		extern __shared__ int temp[];

		int* spotTemp = &temp[sharedMemPerSpot*i];

		float* tmp = (float*)spotTemp;
		float* tmpOut = &tmp[K*K];
		int* P = &spotTemp[2*K*K];

		InvertMatrix(K, tmp, P, tmpOut);

		for (int j=0;j<K;j++)
			d_crlb[K*i + j] = sqrt(abs(tmpOut[j*(K + 1)]));
	}, numspots * sharedMemPerSpot, stream, numThreads);
}


void CUDA_PSF::Estimate(const float * d_sample, const float * d_const, const int* d_roipos, const float * d_initial, 
	float * d_theta, float* d_diag, int* iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream) 
{
	auto buffers = GetDeviceBuffers(stream, numspots);

	float* d_deriv = buffers->derivatives.data();
	float* d_ev = buffers->expectedValue.data();

	// Compute Initial positions
//	if(!d_initial)
	//	InitialEstimate(d_sample, d_const, d_roipos, d_initial, d_theta, numspots);

	for (int i = 0; i < maxiterations; i++)
	{
		Derivatives(d_deriv, d_ev, d_theta, d_const, d_roipos, numspots, stream);

		LaunchKernel(numspots, [=]__device__(int i) {
//			LevMarOptimizer
		}, 0, stream);
	}
}

CUDA_PSF_Wrapper::CUDA_PSF_Wrapper(CUDA_PSF * cudaPSF) : 
	PSF(cudaPSF->SampleSize(), cudaPSF->NumConstants(), cudaPSF->ThetaSize(), cudaPSF->DiagSize(), cudaPSF->ThetaFormat()), psf(cudaPSF)
{}


CUDA_PSF_Wrapper::~CUDA_PSF_Wrapper()
{
	delete psf;
}


void CUDA_PSF_Wrapper::FisherMatrix(const float * h_theta, const float* h_const, const int* spot_pos, int numspots, float * h_FI)
{
	DeviceArray<float> d_theta(numspots*ThetaSize(), h_theta);
	DeviceArray<float> d_FI(numspots*ThetaSize()*ThetaSize());
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	psf->FisherMatrix(d_theta.ptr(), d_const.ptr(), d_roipos.ptr(), numspots, d_FI.ptr(), 0);
	d_FI.CopyToHost(h_FI);
}


void CUDA_PSF_Wrapper::ExpectedValue(float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots)
{
	DeviceArray<float> d_ev(numspots*SampleCount());
	DeviceArray<float> d_theta(numspots*ThetaSize(), h_theta);
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	psf->ExpectedValue(d_ev.ptr(), d_theta.ptr(), d_const.ptr(), d_roipos.ptr(), numspots, 0);
	d_ev.CopyToHost(h_expectedvalue);
}

void CUDA_PSF_Wrapper::Derivatives(float * h_deriv, float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots)
{
	DeviceArray<float> d_theta(numspots*ThetaSize(), h_theta);
	DeviceArray<float> d_ev(numspots*SampleCount());
	DeviceArray<float> d_deriv(numspots*ThetaSize()*psf->SampleCount());
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	psf->Derivatives(d_deriv.ptr(), d_ev.ptr(), d_theta.ptr(), d_const.ptr(), d_roipos.ptr(), numspots, 0);
	d_ev.CopyToHost(h_expectedvalue);
	d_deriv.CopyToHost(h_deriv);
}

void CUDA_PSF_Wrapper::Estimate(const float * h_sample, const float* h_const, const int* spot_pos, const float * h_initial, 
	float * h_theta, float * h_diag, int* h_iterations, int numspots, float* h_trace, int traceBufLen)
{
	DeviceArray<float> d_smp(numspots*SampleCount(), h_sample);
	DeviceArray<float> d_initial(h_initial ? numspots * ThetaSize() : 0, h_initial);
	DeviceArray<float> d_theta(numspots*ThetaSize());
	DeviceArray<float> d_diag(numspots*DiagSize());
	DeviceArray<float> d_trace(numspots*traceBufLen*ThetaSize());
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	DeviceArray<int> d_iterations(numspots);
	psf->Estimate(d_smp.ptr(), d_const.ptr(), d_roipos.ptr(), d_initial.ptr(), d_theta.ptr(), 
		d_diag.ptr(), d_iterations.ptr(), numspots, d_trace.ptr(), traceBufLen, 0);
	d_theta.CopyToHost(h_theta);
	if (h_trace) d_trace.CopyToHost(h_trace);
	if (h_diag) d_diag.CopyToHost(h_diag);
	if (h_iterations) d_iterations.CopyToHost(h_iterations);
}

CenterOfMassEstimator::CenterOfMassEstimator(int roisize) : roisize(roisize), CUDA_PSF({ roisize,roisize }, 0, 4, 0, 0, "x,y,I")
{}

void CenterOfMassEstimator::Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_theta, float* d_diag, int* d_iterations,
	int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  
{
	int numThreads = 1024;
	int roisize = SampleSize()[0];
	Vector4f* estim = (Vector4f*)d_theta;
	LaunchKernel(numspots, [=]__device__(int i) {
		const float* smp = &d_sample[i*roisize*roisize];
		auto com = ComputeCOM(smp, { roisize,roisize });
		estim[i] = { com[0],com[1],com[2],0.0f };
		d_iterations[i] = 0;
	}, 0, stream, numThreads);
}

PhasorEstimator::PhasorEstimator(int roisize) : roisize(roisize), CUDA_PSF({ roisize,roisize }, 0,4, 0, 0, "x,y,I,bg")
{}

void PhasorEstimator::Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, 
	float* d_theta, float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
{
	int roisize = SampleSize()[0];
	Vector4f* estim = (Vector4f*)d_theta;
	LaunchKernel(numspots, [=]__device__(int i) {
		const float* smp = &d_sample[i*roisize*roisize];

		Vector3f e = ComputePhasorEstim(smp, roisize,roisize );
		estim[i] = { e[0],e[1],e[2],0.0f };

		d_iterations[i] = 0;
	}, 0, stream, 1024);
}


CDLL_EXPORT PSF * CreateCenterOfMassEstimator(int roisize, Context* ctx)
{
	auto* p = new CUDA_PSF_Wrapper(new CenterOfMassEstimator(roisize));
	if (ctx) p->SetContext(ctx);
	return p;
}

CDLL_EXPORT PSF* CreatePhasorEstimator(int roisize, Context* ctx)
{
	auto* p = new CUDA_PSF_Wrapper(new PhasorEstimator(roisize));
	if (ctx) p->SetContext(ctx);
	return p;
}

CopyROI_PSF::CopyROI_PSF(CUDA_PSF * org) : 
	CUDA_PSF(org->SampleSize(), org->NumConstants(), org->ThetaSize(), 
		org->SampleCount(), org->MaxIterations(), org->ThetaFormat()), psf(org)
{
}

void CopyROI_PSF::Estimate(const float * d_sample, const float * d_const, const int * d_roipos, const float * d_initial, 
	float * d_theta, float * d_diag, int * d_iterations,  int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
{
	psf->Estimate(d_sample, d_const, d_roipos, d_initial, d_theta, d_diag, d_iterations, numspots, d_trace, traceBufLen, stream);

	int smpcount = SampleCount();
	LaunchKernel(numspots, [=]__device__(int i) {
		for (int j = 0; j < smpcount; j++)
			d_diag[i*smpcount + j] = d_sample[i*smpcount + j];
	} , 0, stream);
}

CDLL_EXPORT PSF * CopyROI_CreatePSF(PSF* model, Context* ctx)
{
	auto* p = new CUDA_PSF_Wrapper(new CopyROI_PSF(model->GetCUDA_PSF()));
	if (ctx) p->SetContext(ctx);
	return p;
}

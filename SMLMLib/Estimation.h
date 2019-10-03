// Model-independent Fisher matrix calculation and Levenberg-Marquardt optimizer (https://www.nature.com/articles/nmeth0510-338)
// All estimators here are for Poisson-distributed samples.
#pragma once

#include "SolveMatrix.h"
#include "Vector.h"
#include <math_constants.h>


template<typename Theta>
struct FisherMatrixType {
	const static int K = Theta::K;
	typedef Vector<typename Theta::TElem, K*K> type;
};

template<typename T>
class SampleOffset_None
{
public:
	PLL_DEVHOST T Get(Int2 samplepos, Int2 roipos) const {
		return 0;
	}
	PLL_DEVHOST T Get(Int3 samplepos, Int3 roipos) const {
		return 0;
	}
};

template<typename T>
class SampleOffset_sCMOS
{
public:
	PLL_DEVHOST T Get(Int2 samplepos, Int2 roipos) const {
		Int2 pos = roipos + samplepos;
		return d_vargain2[pos[0] * pitch + pos[1]];
	}
	PLL_DEVHOST T Get(Int3 samplepos, Int3 roipos) const {
		return d_vargain2[(roipos[1]+samplepos[1]) * pitch + (roipos[2]+samplepos[2])];
	}
	const float* d_vargain2; // var / g^2
	int pitch;
};

template<typename Theta>
struct EstimationResult
{
	template<typename TLevMarResult>
	PLL_DEVHOST EstimationResult(Theta initialValue, TLevMarResult levMarResult, const Theta& crlb, float spotScores=0.0f) :
		estimate(levMarResult.estimate), initialValue(initialValue), crlb(crlb),
		iterations(levMarResult.iterations), spotScores(spotScores) {
	}
	PLL_DEVHOST EstimationResult() { }

	Theta estimate;
	Theta initialValue;
	Theta crlb;
	int iterations;
	Int2 roiPosition;
	float spotScores;
	float loglikelihood; // sum(x log(mu) - mu)
};




template<typename TModel, typename Theta, typename TSampleOffset>
PLL_DEVHOST auto ComputeFisherMatrix(const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset,
	const Theta& theta, typename Theta::TElem* expectedValue=nullptr)
	-> typename FisherMatrixType<Theta>::type
{
	const int K = model.K;
	typedef typename Theta::TElem T;
	typename FisherMatrixType<Theta>::type fi;

	for (int i = 0; i < K*K; i++)
		fi[i] = 0;
	
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T* jacobian) {
		int n = model.SampleIndex(smpIndex);
		mu += offset.Get(smpIndex, roipos);
		if(expectedValue)
			expectedValue[n] = mu;

		float inv_mu_c = 1.0f / (mu > 1e-8f ? mu : 1e-8f);
		for (int i = 0; i < K; i++) {
			for (int j = i; j < K; j++) {
				const T fi_ij = jacobian[i] * jacobian[j] * inv_mu_c;
				fi[K*i + j] += fi_ij;
			}
		};
	}, theta, roipos);
	// fill below diagonal
	for (int i = 1; i < K; i++)
		for (int j = 0; j < i; j++)
			fi[K*i + j] = fi[K*j + i];

	return fi;
}

template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST void ComputeLevMarAlphaBeta(T* __restrict alpha, T* __restrict beta, const typename TModel::Theta& theta,
	const T* sample, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset)
{
	const int K = model.K;
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T* jacobian) {
		float readnoise = offset.Get(smpIndex, roipos);
		int n = model.SampleIndex(smpIndex);
		T sampleValue = sample[n] + readnoise;
		mu += readnoise;
		if (sampleValue < 1e-6f) sampleValue = 1e-6f;

		float mu_c = mu > 1e-6f ? mu : 1e-6f;
		float invmu = 1.0f / mu_c;
		T x_f2 = sampleValue * invmu*invmu;

		for (int i = 0; i < K; i++)
			for (int j = i; j < K; j++)
				alpha[K*i + j] += jacobian[i] * jacobian[j] * x_f2;

		T beta_factor = 1 - sampleValue * invmu;
		for (int i = 0; i < K; i++) {
			beta[i] -= beta_factor * jacobian[i];
		}
	}, theta, roipos);

	// fill below diagonal
	for (int i = 1; i < K; i++)
		for (int j = 0; j < i; j++)
			alpha[K*i + j] = alpha[K*j + i];
}



template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST T ComputeLogLikelihood(const typename TModel::Theta& theta, const T* sample, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset)
{
	T LL = 0;
	const int K = model.K;
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T (&jacobian)[K]) {
		int idx = model.SampleIndex(smpIndex);
		float readnoise = offset.Get(smpIndex, roipos);
		T sampleValue = sample[idx] + readnoise;
		mu += readnoise;
		if(mu>1e-8f)
			LL += sampleValue * log(mu) - mu;
	}, theta, roipos);
	return LL;
}


template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST T ComputeChiMLE(const typename TModel::Theta& theta, const T* sample, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset)
{
	T LL = 0;
	const int K = model.K;
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T (&jacobian)[K]) {
		float readnoise = offset.Get(smpIndex, roipos);
		int idx = model.SampleIndex(smpIndex);
		T sampleValue = sample[idx] + readnoise;
		mu += readnoise;
		if (sampleValue > 1e-8)
			LL += (mu - sampleValue) - sampleValue * log(mu / sampleValue);
	}, theta, roipos);
	return LL;
}


template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST void ComputeExpectedValue(const typename TModel::Theta& theta, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset, T* expectedValue)
{
	const int K = model.K;
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T(&jacobian)[K]) {
		expectedValue[model.SampleIndex(smpIndex)] = mu + offset.Get(smpIndex, roipos);
	}, theta, roipos);
}

template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST void ComputeDerivatives(const typename TModel::Theta& theta, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset, T* derivatives, T* expectedValue = 0)
{
	const int K = model.K;
	const int img_size = model.SampleCount();
	model.ComputeDerivatives([&](const typename TModel::TSampleIndex& smpIndex, T mu, const T(&jacobian)[K]) {
		int j = model.SampleIndex(smpIndex);
		for (int i = 0; i < K; i++) {
			derivatives[j + i * img_size] = jacobian[i];
		}
		if(expectedValue) 
			expectedValue[j] = mu + offset.Get(smpIndex, roipos);
	}, theta, roipos);
}



template<typename TParams >
struct OptimizerResult
{
	TParams estimate;
	int iterations;
	TParams initialValue;
};

template<typename TModel, typename TSampleOffset>
class LevMarOptimizer {
public:
	typedef typename TModel::T T;
	typedef typename TModel::Theta Theta;

	typename TModel::Theta theta, lastTheta;

	PLL_DEVHOST LevMarOptimizer(Theta initialValue) : 
		theta(initialValue), lastTheta(initialValue)
	{}
	PLL_DEVHOST LevMarOptimizer() {}

	PLL_DEVHOST void Init(Theta t) {
		theta = t;
		lastTheta = t;
	}

	PLL_DEVHOST bool Step(const T* sample, int iteration, const TModel& model, const typename TModel::TSampleIndex& roipos, float lambdaStep, TSampleOffset sampleOffset) {
		const int K = model.K;
		T alpha[K*K] = {};
		T LU[K*K] = {};
		T beta[K] = {};
		T thetaStep[K];

		ComputeLevMarAlphaBeta(alpha, beta, theta, sample, model, roipos, sampleOffset);

		for (int k = 0; k < K; k++)
			alpha[k*K + k] *= 1 + lambdaStep;

		if (!Cholesky(K, alpha, LU))
			return false;
		if (!SolveCholesky(LU, beta, thetaStep))
			return false;

		bool smallDeltas = true;
		for (int k = 0; k < K; k++) {
			T delta = theta[k] - lastTheta[k];
			if (abs(delta) > model.StopLimit(k)) {
				smallDeltas = false;
				break;
			}
		}

		lastTheta = theta;
		for (int k = 0; k < K; k++)
			theta[k] += thetaStep[k];
		model.CheckLimits(theta);
		if (iteration > 0 && smallDeltas)
			return false;

		return true;
	}
};


template<typename TModel, typename TSampleOffset>
PLL_DEVHOST OptimizerResult <typename TModel::Theta > LevMarOptimize(const typename TModel::T* sample, typename TModel::Theta initialValue, 
	const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset, int maxIterations, typename TModel::Theta* trace=0, int traceBufLen=0, float startLambdaStep=0.1f)
{
	LevMarOptimizer<TModel, TSampleOffset> optimizer(initialValue);

	int i;
	for (i = 0; i < maxIterations; i++) {
		if (i < traceBufLen)
			trace[i] = optimizer.theta;

		if (!optimizer.Step(sample, i, model, roipos, startLambdaStep, offset))
			break;
	}
	OptimizerResult<typename TModel::Theta> lr;
	lr.iterations = i;
	lr.estimate = optimizer.theta;
	lr.initialValue = initialValue;
	return lr;
}

template<typename T, typename TModel, typename TSampleOffset>
PLL_DEVHOST OptimizerResult<typename TModel::Theta> 
ComputeMLE(const T* sample, const TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &sampleOffset, int maxIterations,
	const typename TModel::Theta *initial=0, typename TModel::Theta* trace=0, int traceBufLen=0, float startLambdaStep=0.1f)
{
	typedef typename TModel::Theta Theta;
	OptimizerResult<Theta> r;
	Theta initialValue;

	if (!initial) {
		initialValue = model.ComputeInitialEstimate(sample, 0, roipos, sampleOffset);
		r = LevMarOptimize(sample, initialValue, model, roipos, sampleOffset, maxIterations,
			trace, traceBufLen, startLambdaStep);

		if (model.NumStartPos > 1) {
			auto bestLikelihood = ComputeLogLikelihood(r.estimate, sample, model, roipos, sampleOffset);
			for (int i = 1; i < model.NumStartPos; i++) {
				Theta iv_new = model.ComputeInitialEstimate(sample, i, roipos, sampleOffset);
				OptimizerResult<Theta> r_new = LevMarOptimize(sample, iv_new, model, roipos, sampleOffset, maxIterations, 
					trace, traceBufLen, startLambdaStep);
				auto ll = ComputeLogLikelihood(r_new.estimate, sample, model, roipos, sampleOffset);
				if (ll > bestLikelihood) {
					bestLikelihood = ll;
					r = r_new;
					initialValue = iv_new;
				}
			}
		}
	}
	else {
		initialValue = *initial;
		r = LevMarOptimize(sample, initialValue, model, roipos, sampleOffset, maxIterations, 
			trace, traceBufLen, startLambdaStep);
	}
	return r;
}

// TModel needs to implement ComputeSecondDerivatives() for this to work
template<typename TModel, typename TSampleOffset>
PLL_DEVHOST OptimizerResult<typename TModel::Theta> NewtonRaphson(const typename TModel::T* __restrict sample, typename TModel::Theta initialValue,
	TModel& model, const typename TModel::TSampleIndex& roipos, const TSampleOffset &offset, int maxIterations, typename TModel::Theta* trace = 0, int traceBufLen = 0, float stepFactor=0.8f)
{
	const int K = TModel::Theta::K;
	typedef typename TModel::T T;
	typename TModel::Theta theta = initialValue, lastTheta;

	T num[K];
	T denom[K];
	int i;
	for (i = 0; i < maxIterations; i++) {
		if (i < traceBufLen)
			trace[i] = theta;

		for (int j = 0; j < K; j++)
			num[j] = denom[j] = 0.0f;

		model.ComputeSecondDerivatives([&](const typename TModel::TSampleIndex smpIndex, T mu, const T(&firstOrder)[K], const T(&secondOrder)[K]) {
			int idx = model.SampleIndex(smpIndex);
			float readnoise = offset.Get(smpIndex, roipos);
			T x = sample[idx] += readnoise;
			mu += readnoise;
			T mu_clamped = mu > 1e-8f ? mu : 1e-8f;
			T invMu = 1.0f / mu_clamped;
			T m = x * invMu - 1;
			for (int j = 0; j < K; j++) {
				num[j] += firstOrder[j] * m;
				denom[j] += secondOrder[j] * m - firstOrder[j] * firstOrder[j] * x*invMu*invMu;
			}
		}, theta, roipos);
		if (i > 0) {
			bool smallDeltas = true;
			for (int k = 0; k < K; k++) {
				T delta = theta[k] - lastTheta[k];
				if (abs(delta) > model.StopLimit(k)) {
					smallDeltas = false;
					break;
				}
			}
			if (smallDeltas)
				break;
		}

		lastTheta = theta;
		typename TModel::Theta step;
		for (int j = 0; j < K; j++)
			step[j] = num[j] / denom[j];

#ifndef __CUDA_ARCH__
		//DebugPrintf("iteration %d: ",i);
		//PrintVector(step);
#endif

		theta -= step*stepFactor;
		model.CheckLimits(theta);
	}
	OptimizerResult<typename TModel::Theta> lr;
	lr.iterations = i;
	lr.estimate = theta;
	return lr;
}


template<typename T, int KK> PLL_DEVHOST Vector<T, CompileTimeSqrt(KK)> ComputeCRLB(const Vector<T, KK>& fisher)
{
	const int K = CompileTimeSqrt(KK);
	Vector<T,KK> inv;
	Vector<T, K> crlb;
	if (InvertMatrix<T,KK>(fisher, inv))
	{
		for (int i = 0; i < K; i++)
			crlb[i] = sqrt(fabs(inv[i*(K + 1)]));
	}
	else {
		for (int i = 0; i < K; i++)
			crlb[i] = INFINITY;
	}
	return crlb;
}


template<typename T>
PLL_DEVHOST Vector3f ComputeCOM(const T* sample, Int2 roisize)
{
	// Compute initial value
	T sumX = 0, sumY = 0, sum = 0;
	for (int y = 0; y < roisize[0]; y++)
		for (int x = 0; x < roisize[1]; x++) {
			T v = sample[y * roisize[1] + x];
			sumX += x * v;
			sumY += y * v;
			sum += v;
		}

	T comx = sumX / sum;
	T comy = sumY / sum;
	return { comx,comy, sum };
}

template<typename T>
PLL_DEVHOST Vector3f ComputePhasorEstim(const T* smp, int w, int h)
{
	//		fx = np.sum(np.sum(roi, 0)*np.exp(-2j*np.pi*np.arange(roi.shape[1]) / roi.shape[1]))
	//		fy = np.sum(np.sum(roi, 1)*np.exp(-2j*np.pi*np.arange(roi.shape[0]) / roi.shape[0]))
	float fx_re = 0.0f, fx_im = 0.0f;
	float freqx = 2 * CUDART_PI_F / w;
	for (int x = 0; x < w; x++)
	{
		float sum = 0.0f;
		for (int y = 0; y < h; y++)
			sum += smp[y*w + x];
		fx_re += sum * cos(-x * freqx);
		fx_im += sum * sin(-x * freqx);
	}
	//    angX = np.angle(fft_values[0,1])
	float angX = atan2(fx_im, fx_re);
	if (angX > 0) angX -= 2 * CUDART_PI_F;
	float posx = abs(angX) / freqx;

	float fy_re = 0.0f, fy_im = 0.0f;
	float freqy = 2 * CUDART_PI_F / h;
	float total = 0.0f;
	for (int y = 0; y < h; y++)
	{
		float sum = 0.0f;
		for (int x = 0; x < w; x++)
			sum += smp[y*w + x];
		fy_re += sum * cos(-y * freqy);
		fy_im += sum * sin(-y * freqy);
		total += sum;
	}

	float angY = atan2(fy_im, fy_re);
	if (angY > 0) angY -= 2 * CUDART_PI_F;
	float posy = abs(angY) / freqy;

	return { posx,posy,total };
}

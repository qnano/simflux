#include "MemLeakDebug.h"
#include "dllmacros.h"
#include "palala.h"
#include "Estimation.h"
#include "StringUtils.h"
#include "MathUtils.h"

#include "ImageProcessingQueue.h"
#include "SpotDetection/SpotDetector.h"
#include "CameraCalibration.h"

#include "GaussianPSF.h"
#include "GaussianPSFModels.h"
#include "SimpleLocalizationTask.h"

#include "PSF.h"
#include "PSFImpl.h"


CDLL_EXPORT PSF* Gauss_CreatePSF_XYIBgSigma(int roisize, bool cuda)
{
	return 0; // model needs to be added
}

CDLL_EXPORT PSF* Gauss_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda)
{
	SampleOffset_None<float> smpofs;

	if (cuda) {
		return new CUDA_PSF_Wrapper(
			new SimpleCalibrationPSF< CUDA_PSFImpl< Gauss2D_Model_XYZIBg<float>, Gauss3D_Calibration, decltype(smpofs) > >
			(calib, smpofs, roisize)
		);
	} else {
		return new SimpleCalibrationPSF< PSFImpl< Gauss2D_Model_XYZIBg<float>, Gauss3D_Calibration, decltype(smpofs) > >
			(calib, smpofs, roisize);
	}
}


CDLL_EXPORT PSF* Gauss_CreatePSF_XYIBg(int roisize, float sigma, bool cuda)
{
	SampleOffset_None<float> smpofs;
	typedef float T;
	
	if (cuda) {
		return new CUDA_PSF_Wrapper(
			new SimpleCalibrationPSF< CUDA_PSFImpl< Gauss2D_Model_XYIBg<T>, T, decltype(smpofs) > >
			(sigma, smpofs, roisize)
		);
	}
	else {
		return new SimpleCalibrationPSF< PSFImpl< Gauss2D_Model_XYIBg<T>, T, decltype(smpofs) > >
			(sigma, smpofs, roisize);
	}
}


CDLL_EXPORT PSF* Gauss_CreatePSF_XYI(int roisize, float sigma, bool cuda)
{
	SampleOffset_None<float> smpofs;

	if (cuda) {
		return new CUDA_PSF_Wrapper(
			new SimpleCalibrationPSF< CUDA_PSFImpl< Gauss2D_Model_XYI<float>, float, decltype(smpofs) > >
			(sigma, smpofs, roisize)
		);
	}
	else {
		return new SimpleCalibrationPSF< PSFImpl< Gauss2D_Model_XYI<float>, float, decltype(smpofs) > >
			(sigma, smpofs, roisize);
	}
}

CDLL_EXPORT void Gauss2D_ComputeMLE(const float *imageData, Gauss2D_EstimationResult* results, int numspots,
	const Vector4f* initialValues, float sigma, int imgw, int maxIterations, float startLambdaStep, bool cuda,
	Vector4f* traces, int traceBufferLength)
{
	//, Gauss2D_EstimationResult* results, Vector4f *traces
	parallel_for_uc(numspots, cuda, PALALA(int i, const float* imageData, const Vector4f * initialValues, Gauss2D_EstimationResult* results, Vector4f *traces) {
		const float* spot_img = &imageData[imgw*imgw*i];

		SampleOffset_None<float> sampleOffset;
		Gauss2D_Model_XYIBg<float> model(imgw, sigma);

		Vector4f initialValue;
		if (!initialValues) {
			auto com = ComputeCOM(spot_img, imgw);
			initialValue = { com[0],com[1],com[2] * 0.9f,com[3] * 0.1f / (imgw*imgw) };
		}
		else
			initialValue = initialValues[i];

		auto r = LevMarOptimize(spot_img, initialValue, model, sampleOffset, maxIterations, &traces[i*traceBufferLength], traceBufferLength, startLambdaStep);
		auto fi = ComputeFisherMatrix(model, sampleOffset, r.estimate);

		results[i] = Gauss2D_EstimationResult(initialValue, r, ComputeCRLB(fi));
	}, const_array(imageData, numspots*imgw*imgw),
		const_array(initialValues, initialValues ? numspots : 0),
		out_array(results, numspots),
		out_array(traces, traceBufferLength*numspots));
}


CDLL_EXPORT void Gauss2D_ComputeFisherMatrix(float* expectedValue, Gauss2D_FisherMatrix * fiMatrix,
	const Vector4f* theta, int numspots, float sigma, int imgw)
{
	for (int i = 0; i < numspots; i++) {
		float* spot_img = expectedValue ? &expectedValue[imgw*imgw*i] : 0;

		Gauss2D_Model_XYIBg<float> model(imgw, sigma);
		SampleOffset_None<float> offset;
		fiMatrix[i] = ComputeFisherMatrix(model, offset, theta[i], spot_img);
	}
}


CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb,
	int numspots, const Vector2f* xy, float sigma, int imgw, int maxIterations, bool cuda)
{
	std::vector<float> psf_space(imgw*imgw*numspots);

	parallel_for_uc(numspots, cuda, PALALA(int i, float* psf_space, const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb, const Vector2f* xy) {
		const float* spot_img = &imageData[imgw*imgw*i];

		float *psf = &psf_space[imgw*imgw*i];
		SampleOffset_None<float> sampleOffset;
		IntensityBgModel<float>::ComputeGaussianPSF(sigma, xy[i][0], xy[i][1], imgw, psf);
		IntensityBgModel<float> model(imgw, psf);

		auto r = LevMarOptimize(spot_img, { 1,0 }, model, sampleOffset, maxIterations);
		//auto r = NewtonRaphson(spot_img, { sum,0 }, model, sampleOffset, maxIterations);
		IBg[i] = r.estimate;
		IBg_crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, sampleOffset, r.estimate));

#ifndef __CUDA_ARCH__
//		DebugPrintf("IBg[%d]={%.4f,%.2f}. x=%.4f, y=%.4f, sum=%.4f\n",
	//		i, r.estimate[0], r.estimate[1], xy[i][0], xy[i][1], sum);
#endif

	}, psf_space,
		const_array(imageData, numspots*imgw*imgw),
		out_array(IBg, numspots),
		out_array(IBg_crlb, numspots),
		const_array(xy, numspots));
}


// Estimate intensity and background from a given image and gaussian center position
CDLL_EXPORT void Gauss2D_IntensityBg_CRLB(Vector2f* IBg_crlb,
	int numspots, const Vector4f* xyIBg, float sigma, int imgw, bool cuda)
{
	std::vector<float> psf_space(imgw*imgw*numspots);

	parallel_for_uc(numspots, cuda, PALALA(int i, float* psf_space, const Vector4f *xyIBg, Vector2f* crlb) {
		float *psf = &psf_space[imgw*imgw*i];
		SampleOffset_None<float> sampleOffset;
		IntensityBgModel<float>::ComputeGaussianPSF(sigma, xyIBg[i][0], xyIBg[i][1], imgw, psf);
		IntensityBgModel<float> model(imgw, psf);

		Vector2f ibg{ xyIBg[i][2], xyIBg[i][3] };
		crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, sampleOffset, ibg));
	}, psf_space,
		const_array(xyIBg, numspots),
		out_array(IBg_crlb, numspots));
}


// spotList [ x y sigmaX sigmaY intensity ]
CDLL_EXPORT void Gauss2D_Draw(double * image, int imgw, int imgh, float * spotList, int nspots, float addSigma)
{
	auto squared = [](double x) { return x * x; };

	for (int i = 0; i < nspots; i++) {
		float* spot = &spotList[5 * i];
		// just a nice heuristic that seems to work well
		double sigmaScale = 2 + log(20000.0f) * 0.1f + addSigma;
		double hwx = spot[2] * sigmaScale;
		double hwy = spot[3] * sigmaScale;
		int minx = int(spot[0] - hwx), miny = int(spot[1] - hwy);
		int maxx = int(spot[0] + hwx + 1), maxy = int(spot[1] + hwy + 1);
		if (minx < 0) minx = 0;
		if (miny < 0) miny = 0;
		if (maxx > imgw - 1) maxx = imgw - 1;
		if (maxy > imgh - 1) maxy = imgh - 1;

		double _1o2sxs = 1.0f / (sqrt(2.0f) * spot[2]);
		double _1o2sys = 1.0f / (sqrt(2.0f) * spot[3]);
		for (int y = miny; y <= maxy; y++) {
			for (int x = minx; x <= maxx; x++) {
				double& pixel = image[y*imgw + x];
				pixel += spot[4] * exp(-(squared((x - spot[0])*_1o2sxs) + squared((y - spot[1])*_1o2sys))) / (2 * MATH_PI*spot[2] * spot[3]);
			}
		}
	}
}



CDLL_EXPORT ImageProcessingQueue<Gauss2D_EstimationResult> * Gauss2D_CreateImageQueue(const ImageQueueConfig & config,
	float sigma, int imgw, int maxSpotsPerFrame, ISpotDetectorFactory* spotDetectorFactory, sCMOS_CalibrationTransform **p_calibTransform)
{
	Gauss2DProcessingQueue* q = new Gauss2DProcessingQueue(config);
	sCMOS_CalibrationTransform *calibTransform = 0;

	auto* spotDetector = spotDetectorFactory->CreateInstance(config, q);
	auto* extractor = new SpotImageExtractorTask(config.cudaStreamCount, *q, *spotDetector, maxSpotsPerFrame, imgw, 1);
	q->AddTask(spotDetector);
	q->AddTask(extractor);
	q->AddTask(new SimpleLocalizationTask<Gauss2D_EstimationResult, Gauss2D_Model_XYIBg<float>, float>
		(config.cudaStreamCount, sigma, extractor, q->resultContainer, calibTransform));

	if (p_calibTransform)
	{
		calibTransform = new sCMOS_CalibrationTransform(config);
		*p_calibTransform = calibTransform;
		q->SetCalibration(calibTransform);
	}

	return q;
}



CDLL_EXPORT ImageProcessingQueue<EstimationResult<Theta_XYZIBg>> * Gauss3D_CreateImageQueue(const ImageQueueConfig & config,
	const Gauss3D_Calibration& calib, int imgw, int maxSpotsPerFrame, ISpotDetectorFactory* spotDetectorFactory, sCMOS_CalibrationTransform **p_calibTransform)
{
	auto* q = new ImageProcessingQueue<EstimationResult<Theta_XYZIBg>>(config);
	sCMOS_CalibrationTransform *calibTransform = 0;

	auto* spotDetector = spotDetectorFactory->CreateInstance(config, q);
	auto* extractor = new SpotImageExtractorTask(config.cudaStreamCount, *q, *spotDetector, maxSpotsPerFrame, imgw, 1);
	q->AddTask(spotDetector);
	q->AddTask(extractor);
	q->AddTask(new SimpleLocalizationTask<EstimationResult<Theta_XYZIBg>, Gauss2D_Model_XYZIBg<float>, Gauss3D_Calibration>
		(config.cudaStreamCount, calib, extractor, q->resultContainer, calibTransform));

	if (p_calibTransform)
	{
		calibTransform = new sCMOS_CalibrationTransform(config);
		*p_calibTransform = calibTransform;
		q->SetCalibration(calibTransform);
	}

	return q;
}


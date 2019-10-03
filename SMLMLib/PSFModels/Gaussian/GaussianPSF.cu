#include "MemLeakDebug.h"
#include "dllmacros.h"
#include "palala.h"
#include "Estimation.h"
#include "StringUtils.h"
#include "MathUtils.h"

#include "CameraCalibration.h"

#include "GaussianPSF.h"
#include "GaussianPSFModels.h"

#include "PSFModels/PSF.h"
#include "PSFModels/PSFImpl.h"



template<typename TModel, typename TSmpOfs> 
PSF* CreateGaussianPSF(int roisize, bool cuda, TModel::Calibration calib, TSmpOfs smpofs, int maxIterations) {
	float levmarInitialAlpha = 0.1f;

	if (cuda) {
		return new CUDA_PSF_Wrapper(
			new SimpleCalibrationPSF< CUDA_PSFImpl< TModel, decltype(calib), decltype(smpofs) > >
			(calib, smpofs, { roisize,roisize }, maxIterations, levmarInitialAlpha)
		);
	}
	else {
		return new SimpleCalibrationPSF< PSFImpl< TModel, decltype(calib), decltype(smpofs) > >
			(calib, smpofs, { roisize,roisize }, maxIterations, levmarInitialAlpha);
	}
}

CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx)
{
	PSF* psf;
	typedef Gauss2D_Model_XYZIBg Model;
	int maxIterations = 50;

	if (sCMOS_calib) {
		psf = CreateGaussianPSF<Model>(roisize, cuda, calib, sCMOS_calib->GetSampleOffset(),maxIterations);
	} else {
		psf = CreateGaussianPSF<Model>(roisize, cuda, calib, SampleOffset_None<float>(),maxIterations);
	}
	if (ctx) psf->SetContext(ctx);
	return psf;
}


CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx)
{
	PSF* psf;
	int maxIterations = 50;

	if (sCMOS_calib) {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBg>(roisize, cuda, { sigmaX,sigmaY }, sCMOS_calib->GetSampleOffset(), maxIterations);
	}
	else {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBg>(roisize, cuda, { sigmaX,sigmaY }, SampleOffset_None<float>(), maxIterations);
	}
	if (ctx) psf->SetContext(ctx);
	return psf;
}


CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx)
{
	PSF* psf;
	int maxIterations = 50;

	if (sCMOS_calib) {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigma>(roisize, cuda, initialSigma, sCMOS_calib->GetSampleOffset(), maxIterations);
	}
	else {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigma>(roisize, cuda, initialSigma, SampleOffset_None<float>(), maxIterations);
	}
	if (ctx) psf->SetContext(ctx);
	return psf;
}



CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx)
{
	PSF* psf;
	int maxIterations = 50;
	Vector2f initialSigma = { initialSigmaX,initialSigmaY };

	if (sCMOS_calib) {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigmaXY>(roisize, cuda, initialSigma, sCMOS_calib->GetSampleOffset(), maxIterations);
	}
	else {
		psf = CreateGaussianPSF<Gauss2D_Model_XYIBgSigmaXY>(roisize, cuda, initialSigma, SampleOffset_None<float>(), maxIterations);
	}
	if (ctx) psf->SetContext(ctx);
	return psf;
}


CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYI(int roisize, float sigma, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx)
{
	PSF* psf;
	int maxIterations = 40;
	typedef Gauss2D_Model_XYI Model;

	if (sCMOS_calib) {
		psf = CreateGaussianPSF<Model>(roisize, cuda, sigma, sCMOS_calib->GetSampleOffset(), maxIterations);
	}
	else {
		psf = CreateGaussianPSF<Model>(roisize, cuda, sigma, SampleOffset_None<float>(), maxIterations);
	}
	if (ctx) psf->SetContext(ctx);
	return psf;
}


CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb,
	int numspots, const Vector2f* xy, const Int2* roipos, const float* sigma, int imgw, int maxIterations, bool cuda)
{
	std::vector<float> psf_space(imgw*imgw*numspots);

	palala_for(numspots, cuda, 
		PALALA(int i, float* psf_space, const Int2* roipos, const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb, const Vector2f* xy, const float* sigma) {
		const float* spot_img = &imageData[imgw*imgw*i];

		float *psf = &psf_space[imgw*imgw*i];
		SampleOffset_None<float> sampleOffset;
		IntensityBgModel::ComputeGaussianPSF({ sigma[i],sigma[i] }, xy[i][0], xy[i][1], imgw, psf);
		IntensityBgModel model({ imgw,imgw }, psf);

		auto r = LevMarOptimize(spot_img, { 1,0 }, model, roipos[i], sampleOffset, maxIterations);
		//auto r = NewtonRaphson(spot_img, { sum,0 }, model, sampleOffset, maxIterations);
		IBg[i] = r.estimate;
		IBg_crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, roipos[i], sampleOffset, r.estimate));
	}, psf_space,
		const_array(roipos, numspots),
		const_array(imageData, numspots*imgw*imgw),
		out_array(IBg, numspots),
		out_array(IBg_crlb, numspots),
		const_array(xy, numspots),
		const_array(sigma, numspots));
}


// Estimate intensity and background from a given image and gaussian center position
CDLL_EXPORT void Gauss2D_IntensityBg_CRLB(Vector2f* IBg_crlb,
	int numspots, const Vector4f* xyIBg, const Int2* roipos, float sigma, int imgw, bool cuda)
{
	std::vector<float> psf_space(imgw*imgw*numspots);

	palala_for(numspots, cuda, PALALA(int i, float* psf_space, const Int2* roipos, const Vector4f *xyIBg, Vector2f* crlb) {
		float *psf = &psf_space[imgw*imgw*i];
		SampleOffset_None<float> sampleOffset;
		IntensityBgModel::ComputeGaussianPSF({sigma, sigma}, xyIBg[i][0], xyIBg[i][1], imgw, psf);
		IntensityBgModel model({ imgw,imgw }, psf);

		Vector2f ibg{ xyIBg[i][2], xyIBg[i][3] };
		crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, roipos[i], sampleOffset, ibg));
	}, psf_space,
		const_array(roipos, numspots),
		const_array(xyIBg, numspots),
		out_array(IBg_crlb, numspots));
}


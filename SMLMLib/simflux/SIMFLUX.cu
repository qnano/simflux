/*
SIMFLUX_BlinkHMM algorithm:

// Sum all frames
// Compute 2D Gaussian (gives invalid intensity but valid position)
// Threshold frames
// Compute intensity + background from thresholded frames
// Improve threshold with HMM
// Compute SILM


*/
#include "MemLeakDebug.h"

#include "palala.h"
#include "SIMFLUX.h"
#include "SolveMatrix.h"

#include "MathUtils.h"
#include "PSFModels/Gaussian/GaussianPSF.h"

#include "Estimation.h"
#include "StringUtils.h"
#include "palala.h"

#include "CameraCalibration.h"

#include "PSFModels/Gaussian/GaussianPSFModels.h"
#include "RandomDistributions.h"

#include "ExcitationModel.h"

#include "FFT.h"

#include "ExcitationModel.h"
#include "SpotDetection/SpotDetector.h"

#include "DebugImageCallback.h"

#pragma warning(disable : 4503) // decorated name length exceeded, name was truncated


#include "SIMFLUX_Models.h"


struct SIMFLUX_SampleIndex
{
	int f, x, y;
};


template<typename T>
PLL_DEVHOST T squared(T x) { return x * x; }

inline PLL_DEVHOST Vector4f Compute2x2Inverse(const Vector4f& m)
{
	float d = 1.0f / (m[0] * m[3] - m[1] * m[2]);
	return { d*m[3], -d * m[1], -d * m[2], d*m[0] };
}

PLL_DEVHOST float NormalDistrProb(float smp, float mu, float variance)
{
	float a = mu - smp;
	return exp(-a * a / (2 * variance)) / sqrt(2 * MATH_PI*variance);
}



PLL_DEVHOST float LogNormalDistrProb(float smp, float mu, float variance)
{
	float a = mu - smp;
	return -a * a / (2 * variance) - log(sqrt(2 * MATH_PI*variance));
}


CDLL_EXPORT void SIMFLUX_ASW_ComputeOnOffProb(const float* rois, const SIMFLUX_Modulation* modulation, Vector4f* gaussFits, 
	Vector2f* IBg, Vector2f* probOnOff, Vector2f* crlbVariance, float* expectedIntensities, const SIMFLUX_ASW_Params& params, int numframes,
	int numspots, const int* startPatterns, const int2* roipos, bool useCuda)
{
	SIMFLUX_ASW_Params p = params;
	std::vector<float> psf_space(p.roisize*p.roisize*numspots);
	int maxIterations = 100;
	int imgsize = p.roisize*p.roisize;

	palala_for(numspots, useCuda,
	PALALA(int i, const float* rois, const SIMFLUX_Modulation* modulation, const int* startPatterns, 
		Vector2f* crlbVariance, float* expectedIntensities, float* psf_space,
		const Vector4f *gaussFits, Vector2f* IBg, Vector2f* probOnOff, const int2* roipos) {
		//int numep, float sigma, const SIMFLUX_Modulation* modulation, int2 roiPos
		float *psf = &psf_space[imgsize*i];
		Vector4f gaussFit = gaussFits[i];
		SampleOffset_None<float> sampleOffset;
		IntensityBgModel::ComputeGaussianPSF({ p.psfSigma,p.psfSigma }, gaussFit[0], gaussFit[1], p.roisize, psf);
		IntensityBgModel IBg_model({ p.roisize,p.roisize }, psf);
		SampleOffset_None<float> smpofs;

		float silmPerPatternBg = gaussFit[3] * (p.numepp / (float)numframes) / p.numepp;
		Vector2f ibg_off{ 1e-4f, silmPerPatternBg };
		float variance_off = Compute2x2Inverse(ComputeFisherMatrix(IBg_model, Int2{}, smpofs, ibg_off))[0];

		FixedFunctionExcitationPatternModel epModel(p.numepp, modulation);
		int e = startPatterns[i];
		for (int f = 0; f < numframes; f++) {
			float q, dqdx, dqdy;
			epModel.ExcitationPattern(q, dqdx, dqdy, e, { gaussFit[0]+roipos[i].x, gaussFit[1]+roipos[i].y });

			const float* spot_img = &rois[imgsize*numframes*i + imgsize * f];
			float sum = 0.0f;
			for (int j = 0; j < imgsize; j++)
				sum += spot_img[j];
			auto r = LevMarOptimize(spot_img, { 1,0 }, IBg_model, Int2{}, sampleOffset, maxIterations);
			//auto r = NewtonRaphson(spot_img, { fmaxf(10, sum-silmPerPatternBg*imgsize),silmPerPatternBg }, IBg_model, sampleOffset, maxIterations);
			IBg[i*numframes + f] = r.estimate;
#ifndef __CUDA_ARCH__
			DebugPrintf("IBg[%d,%d]={%.4f,%.2f}. x=%.4f, y=%.4f, sum=%.4f\n", 
				i,f, r.estimate[0],r.estimate[1], gaussFit[0],gaussFit[1],sum);
#endif

			// Theta_I*Q_f
			Vector2f expected_ibg{ gaussFit[2] * q, silmPerPatternBg };

			float variance_on = Compute2x2Inverse(ComputeFisherMatrix(IBg_model, Int2{}, smpofs, expected_ibg))[0];

			expectedIntensities[i*numframes + f] = expected_ibg[0];
			crlbVariance[i*numframes + f] = { variance_on,variance_off };

			float on_prob = LogNormalDistrProb(r.estimate[0], expected_ibg[0], variance_on);
			float off_prob = LogNormalDistrProb(r.estimate[0], 0, variance_off);
			probOnOff[i*numframes + f] = { on_prob,off_prob};

			e++;
			if (e == p.numepp) e = 0;
		}

	}, const_array(rois, numspots*imgsize*numframes),
		const_array(modulation, p.numepp),
		const_array(startPatterns, numspots),
		out_array(crlbVariance, numspots*numframes),
		out_array(expectedIntensities, numspots*numframes),
		psf_space,
		const_array(gaussFits, numspots),
		out_array(IBg, numspots*numframes),
		out_array(probOnOff, numspots*numframes),
		const_array(roipos, numspots));
//	PALALA(int i, const float* rois, const SIMFLUX_Modulation* modulation, const int* startPatterns, float* psf_space,
	//	const Vector4f *gaussFits, Vector2f* IBg, Vector2f* probOnOff, const Int2* roipos)

}

CDLL_EXPORT void SIMFLUX_ASW_ComputeMLE(const float * imageData, const SIMFLUX_Modulation* modulation,
	Gauss2D_EstimationResult *results, int numspots, int numframes, const SIMFLUX_ASW_Params& SIMFLUX_p,
	const SIMFLUX_Theta* initialValues, const Int3* roipos, int flags, SIMFLUX_Theta* optimizerTrace, int traceBufferLength)
{
	SIMFLUX_ASW_Params p = SIMFLUX_p;
	int imgsize = p.roisize * p.roisize;
	bool cuda = flags & SIMFLUX_MLE_CUDA;

	std::vector<float> imageSums(imgsize*numspots);

	palala_for(numspots, cuda,
		PALALA(int i, const float* imageData, const SIMFLUX_Modulation* mod, Gauss2D_EstimationResult* results, 
			const Int3* roipos, float* imgsums, const SIMFLUX_Theta* initialValues, SIMFLUX_Theta* trace)
	{
		const SIMFLUX_Modulation* spot_modulation = &mod[p.numepp * i];
		const float* spot_img = &imageData[p.roisize*p.roisize*numframes*i];

		SampleOffset_None<float> sampleOffset;
		SIMFLUX_Theta initialValue;
		if (!initialValues) {
			float* sum = &imgsums[imgsize*i];
			Int2 roiposYX{ roipos[i][1],roipos[i][2] };
			ComputeImageSums(spot_img, sum, p.roisize, p.roisize, numframes);
			auto com = ComputeCOM(sum, { p.roisize, p.roisize });
			initialValue = { com[0],com[1],com[2] * 0.9f, 0.0f };

			Gauss2D_Model_XYIBg gaussModel({ p.roisize,p.roisize }, { p.psfSigma,p.psfSigma });
			initialValue = LevMarOptimize(sum, initialValue, gaussModel, roiposYX, sampleOffset, 50).estimate;
		}
		else
			initialValue = initialValues[i];

		FixedFunctionExcitationPatternModel epModel(p.numepp, spot_modulation);
		SIMFLUX_Calibration calib{ epModel, p.psfSigma };
		SIMFLUX_Model model(p.roisize,calib,0,numframes,numframes);

		OptimizerResult<SIMFLUX_Theta> r = LevMarOptimize(spot_img, initialValue, model, roipos[i],
			sampleOffset, p.maxLevMarIterations, &trace[traceBufferLength*i], traceBufferLength, p.levMarLambdaStep);

		auto fi = ComputeFisherMatrix(model, roipos[i], SampleOffset_None<float>(), r.estimate);
		results[i] = Gauss2D_EstimationResult(initialValue, r, ComputeCRLB(fi));

	}, const_array(imageData, numspots*imgsize*numframes),
		const_array(modulation, p.numepp*numspots),
		out_array(results, numspots),
		const_array(roipos, numspots),
		imageSums,
		const_array(initialValues, initialValues ? numspots : 0),
		out_array(optimizerTrace, numspots*traceBufferLength));
}



CDLL_EXPORT void SIMFLUX_ASW_ComputeFisherMatrix(float * expectedValue, Gauss2D_FisherMatrix* fiMatrix, const SIMFLUX_Modulation * mod,
	const SIMFLUX_Theta * theta, const Int3* roipos, int numspots, int nframes, const SIMFLUX_ASW_Params& p)
{
	int nspotpixels = p.roisize*p.roisize*nframes;
	for (int i = 0; i < numspots; i++) {
		auto* spot_mod = &mod[p.numepp*i];

		FixedFunctionExcitationPatternModel epModel(p.numepp, spot_mod);
		SIMFLUX_Calibration calib{ epModel,p.psfSigma };
		SIMFLUX_Model model(p.roisize, calib, 0, nframes, nframes);

		fiMatrix[i] = ComputeFisherMatrix(model, roipos[i], SampleOffset_None<float>(), theta[i], &expectedValue[i*nspotpixels]);
	}
}



CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth, 
	float scale, int numProjAngles, const float *projectionAngles, float* output, float* shifts)
{
	int pw = projectionWidth;
	parallel_for_cpu(numProjAngles, [&](int p) {
		float* proj = &output[pw*p];
		for (int i = 0; i < pw; i++)
			proj[i] = 0.0f;
		float kx = cos(projectionAngles[p]);
		float ky = sin(projectionAngles[p]);

		double moment = 0.0;
		for (int i = 0; i < numpts; i++)
			moment += kx * xyI[i][0] + ky * xyI[i][1];
		float shift = pw / 2 - scale * float(moment) / numpts;
		if(shifts) shifts[p] = shift;

		for (int i = 0; i < numpts; i++) {
			float coord = kx * xyI[i][0] + ky * xyI[i][1];
			int index = int(scale * coord + shift + 0.5f);
			if (index < 0 || index> pw - 1)
				continue;
			proj[index] += xyI[i][2];
		}
	});
}




CDLL_EXPORT void SIMFLUX_DFT2D_Points(const Vector3f* xyI, int numpts, const Vector2f* k, int numk, Vector2f* output, bool useCuda)
{
	palala_for(numk, useCuda, PALALA(int i, const Vector3f* xyI, const Vector2f* k, Vector2f* output) {
		Vector2f k_ = k[i];

		// Use Kahan Sum for reduced errors
		Vector2f sum, c;

		for (int j = 0; j < numpts; j++) {
			float p = xyI[j][0] * k_[0] + xyI[j][1] * k_[1];
			float I = xyI[j][2];
			Vector2f input = { cos(p) * I, sin(p) * I };
			Vector2f y = input - c;
			Vector2f t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
		output[i] = sum;
	}, const_array(xyI, numpts),
		const_array(k, numk),
		out_array(output, numk));
}


struct SpotToExtract {
	int linkedIndex;
	int numroi;
	int firstframe;
};



// Generate a sorted list of ROIs to extract from a tiff file for simflux localization:
// spotToLinkedIdx: int[numspots]
// startframes: int[numlinked]
// ontime: int[numlinked], number of frames the linked spot is on
// result: SpotToExtract[numspots]
// Returns number of elements in result list
CDLL_EXPORT int SIMFLUX_GenerateROIExtractionList(int *startframes, int *ontime, int maxfits, 
								int numlinked, int numpatterns, SpotToExtract* result)
{
	int fits = 0;
	for (int i = 0; i < numlinked; i++) 
	{
		int frame = 0;
		while (frame < ontime[i] && fits<maxfits) {
			int remaining = ontime[i] - frame;

			// Can't use
			if (remaining < numpatterns)
				break;

			if (remaining < numpatterns * 2)
			{
				result[fits].numroi = remaining;
				result[fits].firstframe = startframes[i] + frame;
				result[fits].linkedIndex = i;
				fits++;
				break; // used up all of the linked spot frames
			}
			else {
				result[fits].firstframe = startframes[i] + frame;
				result[fits].numroi = numpatterns;
				result[fits].linkedIndex = i;
				fits++;
				frame += numpatterns;
			}
		}
	}
	return fits;
}


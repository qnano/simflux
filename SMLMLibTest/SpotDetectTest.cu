#include "PSFModels/Gaussian/GaussianPSFModels.h"
#include "SpotDetection/SpotDetector.h"
#include "RandomDistributions.h"
#include "LocalizationQueue.h"
#include "Context.h"
#include "TIFFReadWrite.h"
#include "GetPreciseTime.h"
#include "SpotDetection/GLRTSpotDetector.h"

static std::vector<float> GenerateImageWithSpots(float sigma, int nspots, int w, int h, float bg = 1.0f)
{
	// Generate image
	std::vector<double> image(w*h, bg);
	std::vector<float> spotList(5 * nspots);

	for (int i = 0; i < nspots; i++) {
		float*spot = &spotList[5 * i];

		spot[0] = rand_uniform<float>()*w / 2 + w * .25f;
		spot[1] = rand_uniform<float>()*h / 2 + h * .25f;
		spot[2] = spot[3] = sigma;
		spot[4] = 100;// intensity
	}

	if (nspots > 0)
		Gauss2D_Draw(&image[0], w, h, &spotList[0], nspots);

	for (auto& v : image)
		v = rand_poisson(v);

	return Cast<float, double>(image);
}


PLL_DEVHOST float cdf(float x) {
	return 0.5f*(1 + erf(x / sqrtf(2)));
}

std::vector<IndexWithScore> GLRT(const float* img, int w, int h, int roisize, float sigma, SampleOffset_sCMOS<float> smpofs, float threshold=0.01f)
{
	int wr = w - roisize, hr = h - roisize;
	std::vector<IndexWithScore> output(wr*hr);
	std::vector<float> temprois(wr*hr*roisize*roisize);
	std::vector<float> psf(roisize*roisize);

	IntensityBgModel::ComputeGaussianPSF(sigma, roisize*0.5f, roisize*0.5f, roisize, psf.data());

	palala_for(wr, hr, true, PALALA(int x, int y, const float* img, IndexWithScore* output, float* temproi, const float* psf, const float* vargain2) {
		int i = y * wr + x;
		SampleOffset_sCMOS<float> smpofs_ = smpofs;
		smpofs_.d_vargain2 = vargain2;
		Int2 roipos{ y,x };

		// copy to temporary ROI
		float *roi = &temproi[i * roisize*roisize];
		float sum = 0.0f;
		for (int ry = 0; ry < roisize; ry++)
		{
			for (int rx = 0; rx < roisize; rx++) {
				float v = img[(y + ry)*w + x+rx];
				roi[ry*roisize + rx] = v;
				// assuming this same var/g^2 has already been added before by calibration step (x_i = (D_i-o_i)/g + var/g^2 )
				sum += v - smpofs_.Get({ ry,rx }, roipos); 
			}
		}

		// fit I,bg
		IntensityBgModel h1_model({ roisize,roisize },psf);
		auto h1_r = LevMarOptimize(roi, { 10.0f, 0.0f }, h1_model, Int2{ y,x }, smpofs_, 20);
		auto h1_ll = ComputeLogLikelihood(h1_r.estimate, roi, h1_model, Int2{ y,x }, smpofs_);

		// estimate bg if assuming I=0
		float h0_bg = sum / (roisize*roisize);
		float h0_ll = 0.0f;
		for (int ry=0;ry<roisize;ry++) 
			for (int rx = 0; rx < roisize; rx++) {
				float mu = h0_bg + smpofs_.Get({ ry,rx }, roipos);
				h0_ll += roi[ry*roisize + rx] * log(mu) - mu;
			}

		float Tg = 2.0f*(h1_ll - h0_ll);	// Test statistic (T_g)
		if (Tg < 0.0f) Tg = 0.0f;
		float pfa = 2.0f * cdf(-sqrt(Tg)); // False positive probability (P_fa)

		if (pfa > threshold)
			output[i] = { -1, threshold };
		else 
			output[i] = { i, pfa };

	}, const_array(img, w*h), output, temprois, psf, const_array(smpofs.d_vargain2, smpofs.pitch*h));

/*	std::sort(output.begin(), output.end(), [](const auto & a, const auto & b) -> bool {
		return a.score < b.score;
	});
	*/
	return output;
}

void cudaBreak();


void SpotDetectTest()
{
	float psfSigma = 1.8f;
	auto ctx = std::make_unique<Context>();

	int nspots = 20;
	int w = 400, h = 400;
	auto image = GenerateImageWithSpots(psfSigma, nspots, w,h,5.0f);

	WriteTIFF("spots.tif", &image[0], w,h);

	int roisize = 7;
	float fdr = 0.02f;
	int maxSpots = 10000;
	int nframes = 50;

	std::vector<float> vargain2(w*h, 0.0f);
	SampleOffset_sCMOS<float> smpofs{ vargain2.data(), w };
	std::vector<IndexWithScore> results;
	double t0 = GetPreciseTime();
	for (int i = 0; i < nframes; i++) {
		if ((i % (nframes/10)) == 0)
			DebugPrintf("%d/%d\n", i, nframes);
		results = GLRT(image.data(), w, h, roisize, psfSigma,smpofs, 0.0005f);
	}
	auto pfa = Transform(results, [=](auto r) {return r.score; });

	double t1 = GetPreciseTime();
	float t = t1 - t0;
	DebugPrintf("Processed %d frames (%.1f fps) in %.1f s.\n", nframes, nframes / t, t);

	WriteTIFF("pfa.tiff", pfa.data(), w - roisize, h - roisize);

	ctx.reset(0);
}


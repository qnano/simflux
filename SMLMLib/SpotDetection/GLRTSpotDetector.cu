#include "MemLeakDebug.h"

#include "Vector.h"
#include "CudaUtils.h"
#include "GLRTSpotDetector.h"
#include "SpotDetector.h"
#include "Estimation.h"
#include "MathUtils.h"
#include "DebugImageCallback.h"
#include "PSFModels/Gaussian/GaussianPSFModels.h"

#include <cub/cub.cuh>




GLRTSpotDetector::GLRTSpotDetector(int2 imgsize, GLRTConfig cfg, sCMOS_Calibration* calib): 
	sCMOS_calib(calib), config(cfg)
{
	offset = int2{ cfg.roisize / 2, cfg.roisize / 2 };
	croppedSize = int2{ imgsize.x - cfg.roisize, imgsize.y - cfg.roisize };

	spotImages.Init(croppedSize.x*croppedSize.y*cfg.roisize*cfg.roisize);
	pixelPos.Init(croppedSize.x*croppedSize.y);
	pixelPos_sorted.Init(croppedSize.x*croppedSize.y);
	h_pixelPos.Init(croppedSize.x*croppedSize.y);
	numFoundSpots.Init(1);
	h_numFoundSpots.Init(1);
	std::vector<float> psf(cfg.roisize * cfg.roisize);
	IntensityBgModel::ComputeGaussianPSF({ cfg.psfSigma,cfg.psfSigma }, cfg.roisize * 0.5f, cfg.roisize * 0.5f, cfg.roisize, psf.data());
	this->psf.CopyToDevice(psf);
}

GLRTSpotDetector::~GLRTSpotDetector()
{}


// A model that only fits background. 
struct BgModel : public Gauss2D_PSFModel<float, 1>
{
	PLL_DEVHOST T StopLimit(int k) const
	{
		return 1e-4f;
	}

	PLL_DEVHOST BgModel(int roisize) : Gauss2D_PSFModel({ roisize,roisize }) {}

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
};


__device__ float cdf(float x) {
	return 0.5f*(1 + erf(x / sqrtf(2)));
}

struct non_negative
{
	__host__ __device__
		bool operator()(const IndexWithScore &x)
	{
		return x.index >= 0;
	}
};

template<typename TSmpOfs>
void GLRTSpotDetector::_Detect(const DeviceImage<float>& srcImage, cudaStream_t stream, TSmpOfs smpofs)
{
	float* d_spotimg = spotImages.data();
	int2 croppedSize = this->croppedSize;
	auto cfg = this->config;
	auto offset = this->offset;
	auto src = srcImage.GetConstIndexer();
	const float* d_psf = this->psf.data();
	IndexWithScore* d_pixelpos = pixelPos.data();
	float fppThreshold = config.fppThreshold;

	DeviceArray<float> I(croppedSize.x*croppedSize.y);
	float* I_ = I.data();
	DeviceArray<Vector2f> IBg(croppedSize.x*croppedSize.y);
	Vector2f* IBg_ = IBg.data();
	DeviceArray<float> Bg(croppedSize.x*croppedSize.y);
	float* Bg_ = Bg.data();
	DeviceArray<float> h0_ll(croppedSize.x*croppedSize.y);
	float* h0_ll_ = h0_ll.data();
	DeviceArray<float> h1_ll(croppedSize.x*croppedSize.y);
	float* h1_ll_ = h1_ll.data();
	DeviceArray<float> pfa(croppedSize.x*croppedSize.y);
	float* pfa_ = pfa.data();

//	palala_for(croppedSize.y, croppedSize.x, PALALA(int y,int x,  )

	LaunchKernel(croppedSize.y, croppedSize.x, [=]__device__(int y, int x) {
		int spotIndex = y * croppedSize.x + x;
		float* dst = &d_spotimg[spotIndex*cfg.roisize*cfg.roisize];
		for (int py = 0; py < cfg.roisize; py++)
			for (int px = 0; px < cfg.roisize; px++)
				dst[py*cfg.roisize + px] = src(px + x, py + y);

		IntensityBgModel h1_model({ cfg.roisize,cfg.roisize }, d_psf);
		Int2 roipos{ y,x };
		auto h1_r = LevMarOptimize(dst, Vector2f{ 1.0f,0.0f }, h1_model, roipos, smpofs, 50);
		float h1_ll = ComputeLogLikelihood(h1_r.estimate, dst, h1_model, roipos, smpofs);
		h1_ll_[spotIndex] = h1_ll;
		I_[spotIndex] = h1_r.estimate[0];
		IBg_[spotIndex] = h1_r.estimate;

		
		BgModel h0_model(cfg.roisize);
		/*auto h0_r = LevMarOptimize(dst, Vector<float,1>{ 0.0f }, h0_model, roipos, smpofs, 10);*/

		float sum = 0.0f;
		for (int i = 0; i < cfg.roisize*cfg.roisize; i++) {
			sum += dst[i];
		}
		float h0_bg = sum / (cfg.roisize*cfg.roisize);

		float h0_ll = ComputeLogLikelihood({ h0_bg }, dst, h0_model, roipos, smpofs);
		h0_ll_[spotIndex] = h0_ll;
		Bg_[spotIndex] = h0_bg;

		float Tg = 2.0f*(h1_ll - h0_ll);	// Test statistic (T_g)
		float pfa = 2.0f * cdf(-sqrt(Tg)); // False positive probability (P_fa)
		if (pfa < fppThreshold)
			d_pixelpos[spotIndex] = { (y + offset.y) * croppedSize.x + x + offset.y, pfa };
		else
			d_pixelpos[spotIndex] = { -1, 0.0f };
		pfa_[spotIndex] = pfa;
	}, 0, stream);

	cudaStreamSynchronize(stream);
	auto h0_ll_h = h0_ll.ToVector();
	auto h1_ll_h = h1_ll.ToVector();
	auto I_h = I.ToVector();
	auto pfa_h = pfa.ToVector();
	auto Bg_h = Bg.ToVector();
	auto IBg_h = IBg.ToVector();

	printMatrix(I_h.data(), croppedSize.y, croppedSize.x);
	printMatrix(Bg_h.data(), croppedSize.y, croppedSize.x);
	for (int y = 0; y < croppedSize.y; y++) {
		for (int x = 0; x < croppedSize.x; x++) {
			int i = y * croppedSize.x + x;

			float tg = 2.0f*(h1_ll_h[i] - h0_ll_h[i]);
		}
	}

	ShowDebugImage(croppedSize.x, croppedSize.y, 1, I_h.data(), "Ifit");
	auto spotImg = this->spotImages.ToVector();
	ShowDebugImage(cfg.roisize, cfg.roisize, 10, spotImg.data(), "spotimg");

	if (!partitionBuffer.ptr())
	{
		size_t tempBytes;
		CUDAErrorCheck(cub::DevicePartition::If(0, tempBytes, pixelPos.ptr(), pixelPos_sorted.ptr(),
			numFoundSpots.ptr(), (int)pixelPos.size(), non_negative(), stream));
		partitionBuffer.Init(tempBytes);
	}

	size_t tmpsize = partitionBuffer.size();
	CUDAErrorCheck(cub::DevicePartition::If(partitionBuffer.ptr(), tmpsize, pixelPos.ptr(), pixelPos_sorted.ptr(),
		numFoundSpots.ptr(), (int)pixelPos.size(), non_negative(), stream));
		
	h_pixelPos.CopyFromDevice(pixelPos_sorted, stream);
	h_numFoundSpots.CopyFromDevice(numFoundSpots, stream);
}

void GLRTSpotDetector::Detect(const DeviceImage<float>& srcImage, cudaStream_t stream)
{
	if (sCMOS_calib)
		_Detect(srcImage, stream, sCMOS_calib->GetSampleOffset());
	else
		_Detect(srcImage, stream, SampleOffset_None<float>());
}

SpotLocationList GLRTSpotDetector::GetResults()
{
	auto sl = SpotLocationList();
	sl.d_indices = pixelPos_sorted.data();
	sl.numSpots = h_numFoundSpots[0];
	return sl;
}

void GLRTSpotDetector::Completed()
{
}



class GLRTSpotDetectorFactory : public ISpotDetectorFactory {
public:	
	GLRTSpotDetectorFactory(GLRTConfig cfg, sCMOS_Calibration* calib) :cfg(cfg), calib(calib) {}
	ISpotDetector* CreateInstance(int w, int h) {
		return new GLRTSpotDetector({ w,h }, cfg, calib);
	}
	const char *GetName() { return "GLRT"; }
	GLRTConfig cfg;
	sCMOS_Calibration* calib;
};

CDLL_EXPORT ISpotDetectorFactory * GLRT_Configure(float psfSigma, int maxSpots, float fppThreshold, int roisize, sCMOS_Calibration* calib)
{
	return new GLRTSpotDetectorFactory(GLRTConfig{ psfSigma,fppThreshold,roisize,maxSpots }, calib);
}


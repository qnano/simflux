/*
Spot detection based on "Probability-based particle detection that enables threshold-free and robust in vivo single-molecule tracking" 
Using Generalized Likelihood Ratio Test statistic
*/
#pragma once


#include "SpotDetector.h"
#include "CameraCalibration.h"

struct GLRTConfig {
	float psfSigma, fppThreshold;
	int roisize, maxSpots;
};

class GLRTSpotDetector : public ISpotDetector
{
public:
	GLRTSpotDetector(int2 imgsize, GLRTConfig cfg, sCMOS_Calibration* calib);
	~GLRTSpotDetector();

	// Inherited via ISpotDetector
	virtual void Detect(const DeviceImage<float>& srcImage, cudaStream_t stream) override;
	virtual SpotLocationList GetResults() override;
	virtual void Completed() override;

	template<typename TSmpOfs>
	void _Detect(const DeviceImage<float>& srcImage, cudaStream_t stream, TSmpOfs ofs);

protected:

	GLRTConfig config;
	sCMOS_Calibration *sCMOS_calib;

	int2 offset, croppedSize;
	PinnedArray<float> h_falsePositiveProb; // 
	DeviceArray<int> numFoundSpots;
	PinnedArray<int> h_numFoundSpots; // this needs to be pinned memory, as copying to paged memory is never async.
	DeviceArray<float> spotImages;
	DeviceArray<float> psf;
	DeviceArray<IndexWithScore> pixelPos, pixelPos_sorted;
	PinnedArray<IndexWithScore> h_pixelPos;
	DeviceArray<uint8_t> partitionBuffer;
	DeviceArray<uint8_t> sortBuffer;
};

CDLL_EXPORT ISpotDetectorFactory* GLRT_Configure(float psfSigma, int maxSpots, float fdr, int roisize, sCMOS_Calibration*calib);


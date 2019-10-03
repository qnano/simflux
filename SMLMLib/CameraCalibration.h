// sCMOS camera calibration implemented based on 
// Video-rate nanoscopy using sCMOS camera–specific single-molecule localization algorithms
// https://www.nature.com/articles/nmeth.2488
#pragma once

#include <vector>
#include "Estimation.h"
#include "Context.h"
#include "CudaUtils.h"

class IDeviceImageProcessor : public ContextObject
{
public:
	virtual void ProcessImage(DeviceImage<float>& img, cudaStream_t stream) = 0;
	virtual ~IDeviceImageProcessor() {}
};


class sCMOS_Calibration : public IDeviceImageProcessor
{
public:
	sCMOS_Calibration(int2 imageSize, const float* offset, const float* gain, const float * variance);

	void ProcessImage(DeviceImage<float>& img, cudaStream_t stream);
	SampleOffset_sCMOS<float> GetSampleOffset();

protected:
	DeviceImage<float> d_offset, d_invgain, d_vargain2;
	std::vector<float> h_offset, h_invgain, h_vargain2;
};

class GainOffsetCalibration : public IDeviceImageProcessor {
public:
	GainOffsetCalibration(float gain, float offset) : 
		gain(gain), offset(offset) {}
	// Inherited via ImageProcessor
	virtual void ProcessImage(DeviceImage<float>& img, cudaStream_t stream) override;

	float gain, offset;
};


class GainOffsetImageCalibration : public IDeviceImageProcessor {
public:
	GainOffsetImageCalibration(int2 imgsize, const float* gain, const float* offset);
	// Inherited via ImageProcessor
	virtual void ProcessImage(DeviceImage<float>& img, cudaStream_t stream) override;

	DeviceImage<float> invGain, offset;
};


CDLL_EXPORT sCMOS_Calibration* sCMOS_Calib_Create(int w,int h, const float* offset, const float* gain, const float *variance, Context* ctx);
CDLL_EXPORT GainOffsetCalibration* GainOffsetCalib_Create(float gain, float offset, Context* ctx);


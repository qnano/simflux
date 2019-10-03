#include "MemLeakDebug.h"
#include "CameraCalibration.h"


sCMOS_Calibration::sCMOS_Calibration(int2 imageSize, const float * offset, const float * gain, const float * variance)
	: d_invgain(imageSize.x, imageSize.y),
	d_vargain2(imageSize.x, imageSize.y),
	d_offset(offset, imageSize.x, imageSize.y)
{
	float* h_vargain2 = new float[d_vargain2.NumPixels()];
	for (int i = 0; i < d_vargain2.NumPixels(); i++)
		h_vargain2[i] = variance[i] / (gain[i] * gain[i]);

	d_vargain2.CopyFromHost(h_vargain2);
	delete[] h_vargain2;

	float* invgain = new float[imageSize.x*imageSize.y];
	for (int i=0;i<imageSize.x*imageSize.y;i++)
		invgain[i] = 1.0f / gain[i]; 
	this->d_invgain.CopyFromHost(invgain);
	delete[] invgain;
}

void sCMOS_Calibration::ProcessImage(DeviceImage<float>& sample,cudaStream_t stream)
{
	auto smp = sample.GetIndexer();
	auto offset = this->d_offset.GetIndexer();
	auto invgain = this->d_invgain.GetIndexer();
	auto vargain2 = this->d_vargain2.GetIndexer();

	LaunchKernel(sample.width, sample.height, [=]__device__(int x, int y) {
		smp(x, y) = (smp(x, y) - offset(x, y)) * invgain(x, y);// +vargain2(x, y);  var/gain^2 is now added during optimization
	}, 0, stream);
}




SampleOffset_sCMOS<float> sCMOS_Calibration::GetSampleOffset()
{
	SampleOffset_sCMOS<float> offset;
	offset.d_vargain2 = d_vargain2.data;
	offset.pitch = d_vargain2.PitchInPixels();
	return offset;
}


void GainOffsetCalibration::ProcessImage(DeviceImage<float>& img, cudaStream_t stream)
{
	float invgain = 1.0f / this->gain, offset = this->offset;
	auto indexer = img.GetIndexer();
 	LaunchKernel(img.height, img.width, [=]__device__(int y, int x) {
		float v = (indexer(x, y) - offset) * invgain;
		if (v < 0.0f) v = 0.0f;
		indexer(x, y) = v;
	}, 0, stream);
}



GainOffsetImageCalibration::GainOffsetImageCalibration(int2 imgsize, const float * gain, const float * offset) :
	invGain(imgsize), offset(imgsize)
{
	float* invGain = new float[imgsize.x*imgsize.y];
	for (int i = 0; i < imgsize.x*imgsize.y; i++)
		invGain[i] = 1.0f / gain[i];
	this->invGain.CopyFromHost(invGain);
	this->offset.CopyFromHost(offset);
}

void GainOffsetImageCalibration::ProcessImage(DeviceImage<float>& img, cudaStream_t stream)
{
	auto invGain_ = invGain.GetConstIndexer();
	auto offset_ = offset.GetConstIndexer();
	auto img_ = img.GetIndexer();
	LaunchKernel(img.height, img.width, [=]__device__(int y, int x) {
		float v = (img_(x, y) - offset_(x, y))*invGain_(x, y);
		if (v < 0.0f) v = 0.0f;
		img_(x, y) = v;
	}, 0, stream);
}



CDLL_EXPORT sCMOS_Calibration * sCMOS_Calib_Create(int w, int h, const float * offset, const float * gain, const float * variance, Context* ctx)
{
	auto* r = new sCMOS_Calibration({ w,h }, offset, gain, variance);
	if (ctx) r->SetContext(ctx);
	return r;
}

CDLL_EXPORT GainOffsetCalibration * GainOffsetCalib_Create(float gain, float offset, Context* ctx)
{
	auto* r = new GainOffsetCalibration(gain, offset);
	if (ctx) r->SetContext(ctx);
	return r;
}


CDLL_EXPORT GainOffsetImageCalibration * GainOffsetImageCalib_Create(int width,int height, const float *gain, const float* offset, Context* ctx)
{
	auto* r = new GainOffsetImageCalibration({ width,height },gain, offset);
	if (ctx) r->SetContext(ctx);
	return r;
}

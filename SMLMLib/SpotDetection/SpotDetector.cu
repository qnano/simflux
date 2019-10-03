#include "MemLeakDebug.h"

#include <cassert>
#include "CudaUtils.h"

#include "ImageFilters.h"

#include <thrust/functional.h>
#include <cub/cub.cuh>
#include "Vector.h"
#include "SpotDetector.h"

#include "DebugImageCallback.h"
#include "CameraCalibration.h"

#include <memory>

// DLL API
CDLL_EXPORT ISpotDetectorFactory* SpotDetector_Configure(const SpotDetectorConfig& config)
{
	return new SpotDetector::Factory(config);
}
CDLL_EXPORT void SpotDetector_DestroyFactory(ISpotDetectorFactory* factory)
{
	delete factory;
}




struct non_negative
{
	__host__ __device__
		bool operator()(const IndexWithScore &x)
	{
		return x.index >= 0;
	}
};

SpotDetector::SpotDetector(int2 imgsize, const SpotDetectorConfig & cfg) : config(cfg)
{
	temp.Init(imgsize);
	filtered1.Init(imgsize);
	filtered2.Init(imgsize);
	maxFiltered.Init(imgsize);
	int numpixels = imgsize.x*imgsize.y;
	indices.Init(numpixels);
	selectedIndices.Init(numpixels);
	h_selected.Init(numpixels);
	numFoundSpots.Init(1);
	numspots.Init(1);
}




void SpotDetector::Detect(const DeviceImage<float>& srcImage, cudaStream_t stream)
{
	UniformFilter2D(srcImage, temp, filtered1, config.uniformFilter1Size, stream);
	UniformFilter2D(srcImage, temp, filtered2, config.uniformFilter2Size, stream);

	ApplyBinaryOperator2D(filtered1, filtered2, filtered1, thrust::minus<float>(), stream);
	ComparisonFilter2D(filtered1, temp, maxFiltered, config.maxFilterSize, thrust::maximum<float>(), stream);

	// convert to indices
	float* d_filtered1 = filtered1.data;
	float* d_max = maxFiltered.data;
	int pitch = filtered1.pitch / sizeof(float);
	IndexWithScore *d_indices = indices.ptr();
	int w = filtered1.width;
	int h = filtered1.height;
	float minIntensity = config.minIntensity, maxIntensity = config.maxIntensity;
	int roisize = config.roisize;

	LaunchKernel(w, h, [=]__device__(int x, int y) {
		float a = d_filtered1[y*pitch + x];
		float b = d_max[y*pitch + x];
		bool isMax = a == b
			&& x > roisize / 2
			&& y > roisize / 2
			&& x + (roisize - roisize / 2) < w - 1
			&& y + (roisize - roisize / 2) < h - 1;

		if (isMax && b > minIntensity && b < maxIntensity)
			d_indices[y*w + x] = { y * w + x, b };
		else
			d_indices[y*w + x] = { -1, 0.0f };
	}, 0, stream);

	if (!partitionTempStorage.ptr())
	{
		size_t tempBytes;
		CUDAErrorCheck(cub::DevicePartition::If(0, tempBytes, indices.ptr(), selectedIndices.ptr(), 
			numFoundSpots.ptr(), (int)indices.size(), non_negative(), stream));
		partitionTempStorage.Init(tempBytes);
	}

	size_t tmpsize = partitionTempStorage.size();
	CUDAErrorCheck(cub::DevicePartition::If(partitionTempStorage.ptr(), tmpsize, indices.ptr(),
		selectedIndices.ptr(), numFoundSpots.ptr(), (int)indices.size(), non_negative(), stream));

	h_selected.CopyFromDevice(selectedIndices, stream);
	numFoundSpots.CopyToHost(numspots.data(), true, stream);
}

SpotLocationList SpotDetector::GetResults()
{
	SpotLocationList list;
	list.numSpots = numspots[0];
	list.d_indices = selectedIndices.ptr();

	return list;
}



CDLL_EXPORT int SpotDetector_ProcessFrame(const float* frame, int width, int height, int roisize,
	int maxSpots, float* spotScores, Int2* cornerPosYX, float* rois, ISpotDetectorFactory* sdf, IDeviceImageProcessor* calib) 
{
	try {
		DeviceImage<float> img(width, height);
		img.CopyFromHost((const float*)frame);
		PinnedArray<float> h_img(width*height);

		if (calib) {
			calib->ProcessImage(img, 0);
			img.CopyToHost(h_img.data());
			frame = h_img.data();
			cudaStreamSynchronize(0);
		}

		ISpotDetector* detector = sdf->CreateInstance(width, height);
		detector->Detect(img, 0);
		cudaStreamSynchronize(0);
		detector->Completed();
		auto spotList = detector->GetResults();

		size_t numspots = spotList.numSpots;
		if (numspots > maxSpots) numspots = maxSpots;

		PinnedArray<IndexWithScore> pixelIndices(numspots);
		pixelIndices.CopyFromDevice(spotList.d_indices, numspots, 0);
		delete detector;

		for (int i = 0; i < numspots; i++)
		{
			int centerPixelIndex = pixelIndices[i].index;
			Int2 centerYX = { centerPixelIndex / width, centerPixelIndex % width };
			Int2 corner = centerYX - roisize / 2;
			cornerPosYX[i] = corner;
			spotScores[i] = pixelIndices[i].score;

			for (int y = 0; y < roisize; y++) {
				for (int x = 0; x < roisize; x++) {
					int fy = corner[0] + y, fx = corner[1] + x;
					if (fx < 0) fx = 0;
					if (fy < 0) fy = 0;
					if (fx >= width) fx = width - 1;
					if (fy >= height) fy = height - 1;

					rois[i*roisize*roisize + y * roisize + x] = frame[fy*width + fx];
				}
			}
		}
		return (int)numspots;
	}
	catch (const std::exception& e) {
		DebugPrintf("Exception: %s\n", e.what());
		return 0;
	}

}

CDLL_EXPORT void ExtractROIs(const float *frames, int width, int height, int depth, int roiX, int roiY, int roiZ, 
	const Int3 * startpos, int numspots, float * rois)
{
	for (int i = 0; i < numspots; i++) {
		Int3 pos = startpos[i];
		for (int z = 0; z < roiZ; z++) {
			int fz = pos[0] + z;
			if (fz < 0) fz = 0;
			if (fz >= depth) fz = depth - 1;
			for (int y = 0; y < roiY; y++) {
				int fy = pos[1] + y;
				if (fy >= height) fy = height - 1;
				if (fy < 0) fy = 0;
				for (int x = 0; x < roiX; x++) {
					int fx = pos[2] + x;
					if (fx < 0) fx = 0;
					if (fx >= width) fx = width - 1;

					rois[i*roiX*roiY*roiZ + z * roiX*roiY + y * roiX + x] = frames[z*width*height + fy * width + fx];
				}
			}
		}
	}
}




ISpotDetector * SpotDetector::Factory::CreateInstance(int width, int height)
{
	return new SpotDetector({ width,height }, config);
}


const char * SpotDetector::Factory::GetName()
{
	return "SSA Spot Detector";
}

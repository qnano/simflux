#pragma once

#include "ImageProcessingQueue.h"
#include "SpotDetection/SpotDetector.h"
#include "CameraCalibration.h"


template<typename TResult, typename TModel, typename TCalibration>
class SimpleLocalizationTask : public IComputeTask
{
public:
	typedef TModel::Theta TTheta;

	SimpleLocalizationTask(int nStreams, const TCalibration& calibration, 
		ISpotImageSource* sourceData,
		ResultContainer<TResult>& resultContainer)
		: targetContainer(resultContainer),
		spotImageSource(sourceData),
		maxSpots(sourceData->MaxSpotCount()),
		streamData(nStreams),
		calibration(calibration)
	{
		for (int i = 0; i < nStreams; i++) 
		{
			streamData[i].results.Init(maxSpots);
			streamData[i].h_results.Init(maxSpots);
		}
	}

	ResultContainer<TResult>& targetContainer;
	ISpotImageSource* spotImageSource;
	int maxIterations = 100;
	int maxSpots;
	float startLambdaStep = 0.1f;
	TCalibration calibration;

	struct StreamData
	{
		DeviceArray<TResult> results;
		PinnedArray<TResult> h_results;
		int numSpots;
	};

	std::vector<StreamData> streamData;

	const char* GetTaskName()
	{
		// TODO: Set task name from constructor
		return "Simple Estimation Task";
	}

	template<typename TSampleOffset>
	void EstimateWithModel(IComputeTaskManager* tm, int streamIndex, TSampleOffset offset)
	{
		StreamData& sd = streamData[streamIndex];
		const float* imageData = spotImageSource->GetSpotImages(sd.numSpots, streamIndex);
		TResult* d_results = sd.results.ptr();

		if (sd.numSpots == 0)
			return;

		auto* spotDetections = spotImageSource->GetSpotList(streamIndex).d_indices;

		// VS2017 build tools should support *this lambda capture but this is unsupported in cuda right now.
		// (https://devtalk.nvidia.com/default/topic/1027299/cuda-setup-and-installation/cuda-9-failed-to-support-the-latest-visual-studio-2017-version-15-5/)
		// We now have to capture all of the *this parameters by copying first to a local variable.
		auto startLambdaStep = this->startLambdaStep;
		int maxIterations = this->maxIterations;
		int roisize = spotImageSource->ROISize();

		const int2* roiPos = spotImageSource->GetROICornerPositions(streamIndex);
		TModel model({ roisize,roisize }, calibration);

		LaunchKernel(sd.numSpots, [=] __device__(int i) {
			const float* spot_img = &imageData[roisize*roisize*i];
			Int2 roipos = { roiPos[i].y, roiPos[i].x };
			OptimizerResult<TTheta> r = ComputeMLE(spot_img, model, roipos, offset, maxIterations, 0, 0, 0, startLambdaStep);
			auto fi = ComputeFisherMatrix(model, roipos, offset, r.estimate);
			auto estimationResult = TResult(r.initialValue, r, ComputeCRLB(fi), spotDetections[i].score);

			estimationResult.roiPosition = Int2 { roiPos[i].x, roiPos[i].y };
			estimationResult.loglikelihood = ComputeLogLikelihood(r.estimate, spot_img, model, roipos, offset);

			d_results[i] = estimationResult;
		}, 0, tm->GetStream(streamIndex));

		// copy results
		sd.h_results.CopyFromDevice(sd.results, tm->GetStream(streamIndex));
	}

	bool Execute(IComputeTaskManager* tm, int streamIndex)
	{
		EstimateWithModel(tm, streamIndex, SampleOffset_None<float>());
		return true;
	}

	bool Completed(IComputeTaskManager* tm, int streamIndex)
	{
		auto& sd = this->streamData[streamIndex];

		std::vector<TResult> results(sd.numSpots);

		// copy results to result container
		for (int i = 0; i < sd.numSpots; i++) {
			results[i] = streamData[streamIndex].h_results[i];
		}

		int frame = tm->GetFrameIndex(streamIndex);
		targetContainer.AddResults(frame, std::move(results));
		return true;
	}
};

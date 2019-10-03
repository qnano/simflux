// Image queue that applies calibration correction, gain/offset, uniform filters
#include "ImgFilterQueue.h"
#include "ThreadUtils.h"
#include "ImageFilters.h"
#include "Context.h"

#include "ImgFilterQueueImpl.h"

#include "LocalizationQueue.h"
#include "SpotDetection/SpotDetector.h"
#include "CameraCalibration.h"

#include "CudaMath.h"
#include <thrust/functional.h>

#include "ImageProcessor.h"

// 3D peak finding:
// max(img(x,y,z)) == img(x,y,z)

void TimeFilter(const DeviceImage<float>& srcImages, const DeviceArray<float>& tfilter, DeviceImage<float>& dst, cudaStream_t stream)
{
	assert(srcImages.height == dst.height*(int)tfilter.size());
	assert(srcImages.width == dst.width);

	const float* filter = tfilter.data();
	int timewindow = (int)tfilter.size();
	auto src = srcImages.GetConstIndexer();
	auto dst_ = dst.GetIndexer();
	int h = dst.height;
	LaunchKernel(h, srcImages.width, [=]__device__(int y, int x) {
		float sum = 0.0f;
		for (int t = 0; t < timewindow; t++) {
			sum += filter[t] * src(x, t*h + y);
		}
		dst_(x, y) = sum;
	}, 0, stream);
}



void CastFrame(const DeviceImage<uint16_t>& src, DeviceImage<float>& dst, cudaStream_t stream) 
{
	auto src_ = src.GetConstIndexer();
	auto img_ = dst.GetIndexer();
	LaunchKernel(src.width, src.height, [=]__device__(int x, int y) {
		img_(x, y) = src_(x, y);
	}, 0, stream);
}

TimeFilterQueue::TimeFilterQueue(int w, int h, Config config, Context*ctx) :
	ImgQueue({ w,h }, ctx),
	config(config),
	timeFilterInput(w, h*(int)config.filterKernel.size()),
	timefilter(config.filterKernel)
{}

void TimeFilterQueue::ProcessFrame(std::unique_ptr<FilteredFrame> frame)
{
	frames.push_back(std::move(frame));

	int timewindow = (int)config.filterKernel.size();
	if (frames.size() == config.filterKernel.size())
	{
		// copy all frames to the filter
		for (int i = 0; i < timewindow; i++)
		{
			auto& src = frames[i]->d_filtered;
			timeFilterInput.CopyFromDevice(src.data, src.pitch, 0, i*imgsize.y, imgsize.x, imgsize.y, stream);
		}

		auto result = GetNewOutputFrame();

		// run the time kernel
		TimeFilter(timeFilterInput, timefilter, result->d_filtered, stream);

		// copy the middle frame
		auto& middle = frames[timewindow / 2]->d_image;
		result->framenum = frames[timewindow / 2]->framenum;
		result->d_image.CopyFromDevice(middle.data, middle.pitch, stream);

		cudaStreamSynchronize(stream);

		AddToFinished(std::move(result));

		// remove the last frame
		RecycleInput(std::move(frames.front()));
		frames.pop_front();
	}
}

RawFrameInputQueue::RawFrameInputQueue(Config config, IDeviceImageProcessor * calib, Context*ctx) :
	ImgQueue(config.imageSize, ctx), config(config), calib(calib)
{
	frameNumber = 0;
}

void RawFrameInputQueue::AddHostFrame(uint16_t * data)
{
	std::unique_ptr<RawInputFrame> f = GetNewInputFrame();
	f->framenum = frameNumber++;
	memcpy(f->rawimg.data(), data, sizeof(uint16_t)*config.imageSize.x *config.imageSize.y);

	AddFrame(std::move(f));
}


void RawFrameInputQueue::ProcessFrame(std::unique_ptr<RawInputFrame> frame)
{
	auto output = GetNewOutputFrame();

	frame->d_rawimg.CopyFromHost(frame->rawimg.data(), stream);
	CastFrame(frame->d_rawimg, output->d_image, stream);

	if (calib) {
		calib->ProcessImage(output->d_image, stream);
	}

	output->framenum = frame->framenum;

	cudaStreamSynchronize(stream);
	RecycleInput(std::move(frame));
	AddToFinished(std::move(output));
}


RemoveBackgroundFilterQueue::RemoveBackgroundFilterQueue(int2 imgsize, Int2 filterSize, Context*ctx) :
	ImgQueue(imgsize, ctx), 
	uniformFilterSize(filterSize), 
	temp1(imgsize), 
	d_xyfiltered1(imgsize), 
	d_xyfiltered2(imgsize)
{
}

void RemoveBackgroundFilterQueue::ProcessFrame(std::unique_ptr<Frame> frame)
{
	auto output = GetNewOutputFrame();
	output->framenum = frame->framenum;
	UniformFilter2D(frame->d_image, temp1, d_xyfiltered1, uniformFilterSize[0], stream);
	UniformFilter2D(frame->d_image, temp1, d_xyfiltered2, uniformFilterSize[1], stream);
	ApplyBinaryOperator2D(d_xyfiltered1, d_xyfiltered2, output->d_filtered, thrust::minus<float>(), stream);
	output->d_image.Swap(frame->d_image);

	cudaStreamSynchronize(stream);
	AddToFinished(std::move(output));
	RecycleInput(std::move(frame));
}


class SpotDetectedFrame
{
public:
	SpotDetectedFrame(int2 imgsize) {}
};



struct Empty { 	
	Empty(int2) {}
};


struct SummedFrame : public Frame {
	SummedFrame(int2 imgsize) :
		Frame(imgsize)
	{
		assert(0); // this one should not be called
	}
	SummedFrame(int2 imgsize, int sumframes) :
		Frame(imgsize), sumframes(sumframes), original({ imgsize.x, imgsize.y*sumframes }) {}

	int sumframes;
	DeviceImage<float> original;
};


// TODO: Come up with better class naming..
class SpotDetectorImgQueue : public ImgQueue<SummedFrame, Empty> {
public:
	SpotDetectorImgQueue(int2 imgsize, LocalizationQueue* lq, ISpotDetectorFactory* sd) : 
		ImgQueue(imgsize, lq->GetContext()), h_indices(imgsize.x*imgsize.y), dst(lq) {
		spotDetector = sd->CreateInstance(imgsize.x, imgsize.y);

		auto* psf = lq->GetPSF();
		auto smpsize = psf->SampleSize();

		// The number of dimensions in the psf sample tells us if the PSF uses summed frames (2) or the original frames (3)
		if (smpsize.size() == 2) {
			roiWidth = smpsize[1];
			roiHeight = smpsize[0];
			roiFrames = 1;
		}
		else {
			assert(smpsize.size() == 3);
			roiWidth = smpsize[2];
			roiHeight = smpsize[1];
			roiFrames = smpsize[0];
		}
		h_image.Init(imgsize.x*imgsize.y*roiFrames);
		h_samples.resize(roiWidth*roiHeight*roiFrames);
	}
	~SpotDetectorImgQueue() {
		delete spotDetector;
	}

	int NumFinishedFrames()
	{
		return LockedFunction(framesFinishedMutex, [&]() {return framesFinished; });
	}

protected:
	LocalizationQueue* dst;
	PinnedArray<float> h_image;
	PinnedArray<IndexWithScore> h_indices;
	ISpotDetector* spotDetector;
	std::vector<float> h_samples; 
	int roiWidth, roiHeight, roiFrames;

	int framesFinished=0;
	std::mutex framesFinishedMutex;

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<SummedFrame> frame) override
	{
		auto& samplesize = dst->GetPSF()->SampleSize();

		if (samplesize.size() == 3)
			frame->original.CopyToHost(h_image.data(), stream);
		else
			frame->d_image.CopyToHost(h_image.data(), stream);
		spotDetector->Detect(frame->d_image, stream);

		cudaStreamSynchronize(stream);
		int framenum = frame->framenum;
		RecycleInput(std::move(frame));

		spotDetector->Completed();
		auto results = spotDetector->GetResults();
		h_indices.CopyFromDevice(results.d_indices, results.numSpots, stream);
		cudaStreamSynchronize(stream);

		for (int i = 0; i < results.numSpots; i++)
		{
			int centerPixelIndex = h_indices[i].index;
			int centerY = centerPixelIndex / imgsize.x, centerX = centerPixelIndex % imgsize.x;
			int cornerX = centerX - roiWidth / 2 - 1, cornerY = centerY - roiHeight / 2 - 1;
			int roiPixels = roiWidth * roiHeight;
			for (int z = 0; z < roiFrames; z++) {
				for (int y = 0; y < roiHeight; y++) {
					int fy = cornerY + y;
					if (fy < 0) fy = 0;
					if (fy >= imgsize.y) fy = imgsize.y - 1;
					for (int x = 0; x < roiWidth; x++) {
						int fx = cornerX + x;
						if (fx < 0) fx = 0;
						if (fx >= imgsize.x) fx = imgsize.x - 1;

						h_samples[z * roiPixels + y * roiWidth + x] = h_image[ z * imgsize.x*imgsize.y + fy* imgsize.x + fx];
					}
				}
			}

			if (samplesize.size() == 2) {
				int corner[] = { cornerY,cornerX };
				dst->Schedule(framenum, h_samples.data(), 0, corner);
			}
			else {
				int corner[] = { 0, cornerY,cornerX };
				dst->Schedule(framenum, h_samples.data(), 0, corner);
			}
		}

		LockedAction(framesFinishedMutex, [&]() {
			framesFinished++;
		});
	}
};

class SumFrames : public ImgQueue<Frame, SummedFrame> {
public:
	int count=0;
	int sumframes;
	std::unique_ptr<SummedFrame> current;
	int framenum = 0;

	SumFrames(int2 imgsize, int sumframes, Context* ctx) : ImgQueue(imgsize,ctx), sumframes(sumframes)
	{
		current = GetNewOutputFrame();
	}

	virtual SummedFrame* AllocateNewOutputFrame() override
	{
		return new SummedFrame(imgsize, sumframes);
	}

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<Frame> frame) override
	{
		// Keep original as well as the summed image
		current->original.CopyFromDevice(frame->d_image.data, frame->d_image.pitch, 0, count*imgsize.y, imgsize.x, imgsize.y, stream);

		auto& sum = current->d_image;
		if (count == 0)
			sum.Swap(frame->d_image);
		else {
			sum.Apply(frame->d_image, thrust::plus<float>(), stream);
			cudaStreamSynchronize(stream);
		}
		count++;

		if (count == sumframes)
		{
			current->framenum = framenum++;

			AddToFinished(std::move(current));
			current = GetNewOutputFrame();
			count = 0;
		}
		RecycleInput(std::move(frame));
	}
};

struct SpotLocalizerQueue : public ImageProcessor
{
	class Dispatch : public ImgQueue<SummedFrame,Empty> {
	public:
		Dispatch(int2 imgsize, SpotLocalizerQueue* q, Context* ctx) : ImgQueue(imgsize, ctx), owner(q) {}
 		SpotLocalizerQueue* owner;
		int current=0;

		// Inherited via ImgQueue
		virtual void ProcessFrame(std::unique_ptr<SummedFrame> frame) override
		{
			owner->spotDetectors[current++]->AddFrame(std::move(frame));
			current = current % owner->spotDetectors.size();
		}

		virtual std::unique_ptr<SummedFrame> GetRecycledInputFrame() override
		{
			for (auto& sd : owner->spotDetectors) {
				auto f = sd->GetRecycledInputFrame();
				if (f) return std::move(f);
			}
			return std::unique_ptr<SummedFrame>();
		}
	};

	SpotLocalizerQueue(int2 imgsize, IDeviceImageProcessor* calibration, ISpotDetectorFactory* detectorFactory, 
		LocalizationQueue* localizationQueue, int sumframes, int numThreads)
	{
		this->localizationQueue = localizationQueue;
		this->sumframes = sumframes;

		for (int i = 0; i < numThreads; i++)
			spotDetectors.push_back(std::make_unique<SpotDetectorImgQueue>(imgsize, localizationQueue, detectorFactory));

		RawFrameInputQueue::Config config{ imgsize };
		rawInputQueue = std::make_unique<RawFrameInputQueue>(config, calibration, localizationQueue->GetContext());
		dispatchQueue = std::make_unique<Dispatch>(imgsize, this, localizationQueue->GetContext());

		this->sumFramesQueue = std::make_unique<SumFrames>(imgsize, sumframes, localizationQueue->GetContext());
		rawInputQueue->SetTarget(sumFramesQueue.get());
		this->sumFramesQueue->SetTarget(dispatchQueue.get());
	}

	~SpotLocalizerQueue()
	{
		// stop things before they get deleted to prevent invalid access
		if (sumframes) sumFramesQueue->Stop();
		dispatchQueue->Stop();
		rawInputQueue->Stop();
		for (auto& s : spotDetectors)
			s->Stop();
	}

	int sumframes;
	std::unique_ptr<RawFrameInputQueue> rawInputQueue;
	std::unique_ptr<Dispatch> dispatchQueue;
	std::unique_ptr<SumFrames> sumFramesQueue;
	LocalizationQueue* localizationQueue;
	std::vector<std::unique_ptr<SpotDetectorImgQueue> > spotDetectors;

	// Inherited via ImageProcessor
	virtual void AddFrame(uint16_t * data) override
	{
		rawInputQueue->AddHostFrame(data);
	}
	virtual int GetQueueLength() override
	{
		return rawInputQueue->GetQueueLength();
	}

	virtual int NumFinishedFrames() override
	{
		int total = 0;
		for (auto& sd : spotDetectors)
			total += sd->NumFinishedFrames();
		return total;
	}

	virtual bool IsIdle() override
	{
		bool idle = rawInputQueue->IsIdle() && dispatchQueue->IsIdle() && sumFramesQueue->IsIdle();
		for (auto& sd : spotDetectors)
			idle = idle && sd->IsIdle();
		return idle;
	}
};

CDLL_EXPORT ImageProcessor * SpotLocalizerQueue_Create(int width, int height, LocalizationQueue* queue,
	ISpotDetectorFactory* spotDetectorFactory, IDeviceImageProcessor* preprocessor, 
	int numDetectionThreads, int sumframes, Context* ctx)
{
	int2 imgsize{ width,height };
	SpotLocalizerQueue* q = new SpotLocalizerQueue(imgsize, preprocessor, spotDetectorFactory,
		queue, sumframes, numDetectionThreads);
	if (ctx) q->SetContext(ctx);
	return q;
}

//CDLL_EXPORT ImageProcessor* SelectedSpotLocalizerQueue




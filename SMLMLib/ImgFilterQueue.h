/*

*/
#pragma once
#include <thread>
#include <list>
#include <mutex>
#include <unordered_map>
#include <deque>
#include "CudaUtils.h"
#include "Context.h"
#include "ThreadUtils.h"
#include "Vector.h"

template<typename T>
class IFrameSource {
public:
	virtual bool HasFinishedFrame() = 0;
	virtual std::unique_ptr<T> GetFinishedFrame() = 0;
	virtual void RecycleOutput(std::unique_ptr<T> frame) = 0;
};

template<typename T>
class IFrameSink {
public:
	virtual std::unique_ptr<T> GetRecycledInputFrame() = 0;
	virtual void AddFrame(std::unique_ptr<T> frame) = 0;
};

template<typename TInputFrame, typename TOutputFrame>
class ImgQueue : public ContextObject, public IFrameSource<TOutputFrame>, public IFrameSink<TInputFrame> {
public:
	virtual void ProcessFrame(std::unique_ptr<TInputFrame> frame) = 0;
	void AddToFinished(std::unique_ptr<TOutputFrame> frame);
	ImgQueue(int2 imgsize, Context* ctx);
	~ImgQueue();
	std::unique_ptr<TOutputFrame> GetFinishedFrame();
	void RecycleInput(std::unique_ptr<TInputFrame> f);
	std::unique_ptr<TInputFrame> GetRecycledInputFrame() override; // return null if none
	void RecycleOutput(std::unique_ptr<TOutputFrame> frame) override;
	int NumFinishedFrames();
	bool HasFinishedFrame() override;
	int GetQueueLength();
	void AddFrame(std::unique_ptr<TInputFrame> frame);
	virtual std::unique_ptr<TOutputFrame> GetNewOutputFrame();
	virtual std::unique_ptr<TInputFrame> GetNewInputFrame();

	virtual TOutputFrame* AllocateNewOutputFrame()
	{
		return new TOutputFrame(imgsize);
	}
	virtual TInputFrame* AllocateNewInputFrame()
	{
		return new TInputFrame(imgsize);
	}

	void Stop();
	void SetSource(IFrameSource<TInputFrame>* src) { source = src; }
	void SetTarget(IFrameSink<TOutputFrame>* dst) { target = dst; }

	bool IsIdle();

protected:
	void ProcessThreadMain();

	int2 imgsize;
	IFrameSink<TOutputFrame>* target = 0;
	IFrameSource<TInputFrame>* source = 0;
	std::unique_ptr<std::thread> processThread;
	std::atomic<bool> processingFrame;

	std::deque<std::unique_ptr<TInputFrame>> todo;
	std::mutex todoMutex;

	std::list<std::unique_ptr<TInputFrame>> todoRecycle;
	std::mutex todoRecycleMutex;

	// finished and read out
	std::list<std::unique_ptr<TOutputFrame>> recycle;
	std::mutex recycleMutex;

	// finished but not read-out
	std::list<std::unique_ptr<TOutputFrame>> finished;
	std::mutex finishedMutex;

	int maxQueueLength = 12;

	volatile bool abortProcessing = false;

	cudaStream_t stream;
};


struct RawInputFrame {
	RawInputFrame(int2 imgsize) : 
		rawimg(imgsize.x*imgsize.y), d_rawimg(imgsize.x,imgsize.y), framenum(0) {}
	int framenum;
	PinnedArray<uint16_t> rawimg;
	DeviceImage<uint16_t> d_rawimg;
};


struct Frame {
	Frame(int2 imgsize) : d_image(imgsize) {}
	int framenum = 0;
	DeviceImage<float> d_image; // after calib
};

struct FilteredFrame : public Frame {
	FilteredFrame(int2 imgsize) : Frame(imgsize), d_filtered(imgsize) {}
	DeviceImage<float> d_filtered;
};

class IDeviceImageProcessor;

class RawFrameInputQueue : public ImgQueue<RawInputFrame,Frame> {
public:
	struct Config
	{
		int2 imageSize;
	};
	Config config;
	IDeviceImageProcessor* calib;
	std::atomic<int> frameNumber;

	RawFrameInputQueue(Config config, IDeviceImageProcessor * calib, Context*ctx);
	void AddHostFrame(uint16_t* data);

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<RawInputFrame> frame) override;
};

struct HostMemoryFrame 
{
	HostMemoryFrame(int2 imgsize) : h_image(imgsize.x*imgsize.y), h_filtered(imgsize.x*imgsize.y) {}
	int framenum = 0;
	PinnedArray<float> h_image, h_filtered;
};


class RemoveBackgroundFilterQueue : public ImgQueue<Frame, FilteredFrame> {
public:
	DeviceImage<float> d_xyfiltered1, d_xyfiltered2, temp1;
	Int2 uniformFilterSize;

	RemoveBackgroundFilterQueue(int2 imgsize, Int2 filterSize, Context*ctx);
	virtual void ProcessFrame(std::unique_ptr<Frame> frame) override;
};


class TimeFilterQueue  : public ImgQueue<FilteredFrame, FilteredFrame> {
public:
	struct Config
	{
		std::vector<float> filterKernel;
	};

	TimeFilterQueue(int w, int h, Config config, Context*ctx);
	
protected:
	Config config;
	DeviceImage<float> timeFilterInput; // [timewindow,h,w]
	DeviceArray<float> timefilter;
	std::deque<std::unique_ptr<FilteredFrame>> frames;

	virtual void ProcessFrame(std::unique_ptr<FilteredFrame> frame) override;
};



template<typename TFrame>
class CopyToHostQueue : public ImgQueue<TFrame, HostMemoryFrame> {
public:
	CopyToHostQueue(int2 imgsize, Context*ctx) : ImgQueue(imgsize, ctx) {
		ThrowIfCUDAError(cudaStreamCreate(&stream2));
	}
	~CopyToHostQueue() {
		cudaStreamDestroy(stream2);
	}
	int CopyOutputFrame(float * image, float * filtered);
	virtual void ProcessFrame(std::unique_ptr<TFrame> frame) override;
protected:
	cudaStream_t stream2;
};


void TimeFilter(const DeviceImage<float>& srcImages, const DeviceArray<float>& tfilter, DeviceImage<float>& dst, cudaStream_t stream);

void CastFrame(const DeviceImage<uint16_t>& src, DeviceImage<float>& dst, cudaStream_t stream, float offset, float gain);

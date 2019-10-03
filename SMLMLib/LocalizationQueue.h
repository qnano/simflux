/*
LocalizationQueue runs Estimate and Fisher matrix calculation on a CUDA_PSF object using multiple cuda streams.
*/
#pragma once

#include "CudaUtils.h"
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include "Context.h"

#include "PSFModels/PSF.h"

class CUDA_PSF;
class PSF;

// Accepts host- or device memory localization jobs

class LocalizationQueue : public ContextObject {
public:
	DLL_EXPORT LocalizationQueue(CUDA_PSF* psf, int batchSize, int maxQueueLen, int numStreams, Context* ctx=0);
	DLL_EXPORT ~LocalizationQueue();

	DLL_EXPORT void Schedule(int id, const float* h_samples, 
					const float* h_constants, const int* h_roipos);

	// Schedule many
	DLL_EXPORT void Schedule(int count, const int *id, const float* h_samples,
		const float* h_constants, const int* h_roipos);

	DLL_EXPORT void Flush();
	DLL_EXPORT bool IsIdle();

	DLL_EXPORT int GetResultCount();
	DLL_EXPORT int GetQueueLength(); // in spots

	// Returns the number of actual returned localizations. 
	// Results are removed from the queue after copyInProgress to the provided memory
	DLL_EXPORT int GetResults(int maxresults, float* estim, float* diag, int* iterations, float* likelihoods, float *fi, float* crlb, int* roipos, int *ids);

	DLL_EXPORT CUDA_PSF* GetPSF() { return psf; }

protected:

	struct Batch
	{
		Batch(int maxspots, CUDA_PSF* psf);

		std::vector<int> ids;
		PinnedArray<float> samples, constants;
		PinnedArray<int> roipos;
		PinnedArray<float> estimates, diagnostics;
		PinnedArray<float> fisherInfo, crlb;
		PinnedArray<int> iterations;
		PinnedArray<float> ll;
		int numspots;
	};


	struct StreamData {
		StreamData() { }
		std::unique_ptr<Batch> currentBatch;
		cudaStream_t stream;

		DeviceArray<float> estimates, samples, constants, diagnostics, fi, crlb;
		DeviceArray<int> roipos; // [SampleIndexDims * batchsize]
		DeviceArray<int> iterations;
		DeviceArray<float> ll;
	};

	std::list<std::unique_ptr<Batch>> todo;
	int numActive; // also guarded using todoMutex
	std::mutex todoMutex;

	std::list<std::unique_ptr<Batch>> recycle;
	std::mutex recycleMutex;

	std::list<std::unique_ptr<Batch>> finished;
	std::mutex finishedMutex;

	std::unique_ptr<Batch> next; // currently filling up this batch
	std::mutex scheduleMutex;

	CUDA_PSF* psf;
	std::vector<StreamData> streams;
	int maxQueueLen, batchSize;

	std::thread* thread;
	volatile bool stopThread;

	virtual void Launch(std::unique_ptr<Batch> b, StreamData& sd);
	void ThreadMain();

	int numconst, K, smpcount, smpdims;
};

// C API wrappers
CDLL_EXPORT LocalizationQueue* PSF_Queue_Create(PSF* psf, int batchSize, int maxQueueLen, int numStreams, Context* ctx);
CDLL_EXPORT void PSF_Queue_Delete(LocalizationQueue* queue);

CDLL_EXPORT void PSF_Queue_Schedule(LocalizationQueue* q, int numspots, const int *ids, const float* h_samples,
	const float* h_constants, const int* h_roipos);

CDLL_EXPORT void PSF_Queue_Flush(LocalizationQueue* q);
CDLL_EXPORT bool PSF_Queue_IsIdle(LocalizationQueue* q);

CDLL_EXPORT int PSF_Queue_GetResultCount(LocalizationQueue* q);

// Returns the number of actual returned localizations. 
// Results are removed from the queue after copyInProgress to the provided memory
CDLL_EXPORT int PSF_Queue_GetResults(LocalizationQueue* q, int maxresults, float* estim, float* diag, 
	int * iterations, float* ll, float *fi,  float* crlb, int* roipos, int*ids);

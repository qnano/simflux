#include "LocalizationQueue.h"
#include "PSFModels/PSF.h"
#include "ThreadUtils.h"

LocalizationQueue::LocalizationQueue(CUDA_PSF * psf, int batchSize, int maxQueueLen, int numStreams, Context* ctx) 
	: ContextObject(ctx), maxQueueLen(maxQueueLen), batchSize(batchSize), numconst(psf->NumConstants()), 
	K(psf->ThetaSize()), smpcount(psf->SampleCount()), smpdims(psf->SampleIndexDims()), psf(psf)
{
	if (numStreams < 0)
		numStreams = 3;

	psf->SetMaxSpots(batchSize);

	streams.resize(numStreams);
	for (int i = 0; i < numStreams; i++) {
		auto& sd = streams[i];
		cudaStreamCreate(&sd.stream);

		sd.constants.Init(batchSize*numconst);
		sd.diagnostics.Init(batchSize*psf->DiagSize());
		sd.estimates.Init(batchSize*K);
		sd.roipos.Init(batchSize*smpdims);
		sd.samples.Init(batchSize*smpcount);
		sd.fi.Init(batchSize*K*K);
		sd.crlb.Init(batchSize*K);
		sd.ll.Init(batchSize);
		sd.iterations.Init(batchSize);
	}

	numActive = 0;
	next = std::make_unique<Batch>(batchSize, psf);

	stopThread = false;
	thread = new std::thread([&]() {
		ThreadMain();
	});
}


LocalizationQueue::~LocalizationQueue()
{
	stopThread = true;
	thread->join();

	for (auto& s : streams)
		cudaStreamDestroy(s.stream);
}

void LocalizationQueue::ThreadMain()
{
	if(context)
		cudaSetDevice(context->GetDeviceIndex());

	while (!stopThread)
	{
		bool idle = true;
		for (auto& s : streams) {
			if (!s.currentBatch)
			{
				std::unique_ptr<Batch> fr;
				LockedAction(todoMutex, [&]()
				{
					if (!todo.empty()) {
						fr = std::move(todo.front());
						todo.pop_front();
						numActive++;
					}
				});
				if (fr) {
					Launch(std::move(fr), s);
					idle = false;
				}
			}

			if (s.currentBatch && cudaStreamQuery(s.stream) == cudaSuccess) {
				// all done, move to finished
				LockedAction(finishedMutex, [&]() {
					finished.push_back(std::move(s.currentBatch));
				});
				LockedAction(todoMutex, [&]() {numActive--; });
				idle = false;
			}
		}

		if (idle) // dont waste cpu
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

bool LocalizationQueue::IsIdle()
{
	return LockedFunction(todoMutex, [&]() {
		return todo.size() + numActive;
	}) == 0;
}

int LocalizationQueue::GetResultCount()
{
	return LockedFunction(finishedMutex, [&]() {
		int total = 0;
		for (auto& l : finished)
			total += l->numspots;
		return total;
	});
}

int LocalizationQueue::GetResults(int count, float * estim, float * diag, int* iterations, 
									float* likelihoods, float * fi, float* crlb, int* roipos, int *ids)
{
	int copied = 0;

	while (copied < count) {
		int space = count - copied;

		// Is there batch finished and does it fully fit in the remaining space?
		auto b = LockedFunction(finishedMutex, [&]() {
			if (finished.empty()  || finished.back()->numspots > space)
				return std::unique_ptr<Batch>();

			auto b = std::move(finished.back());
			finished.pop_back();
			return b;
		});

		if (!b)
			return copied;

		if (ids) {
			for (int i = 0; i < b->numspots; i++)
				ids[copied + i] = b->ids[i];
		}

		for (int i = 0; i < K * b->numspots; i++) 
			estim[copied * K + i] = b->estimates[i];
		if (fi) {
			for (int i = 0; i < K*K * b->numspots; i++)
				fi[copied * K*K + i] = b->fisherInfo[i];
		}
		if (crlb) {
			for (int i = 0; i < K*b->numspots; i++)
				crlb[copied*K + i] = b->crlb[i];
		}
		if (roipos) {
			for (int i = 0; i < psf->SampleIndexDims() * b->numspots; i++)
				roipos[copied * psf->SampleIndexDims() + i] = b->roipos[i];
		}
		if (diag) {
			for (int i = 0; i < psf->DiagSize() * b->numspots; i++)
				diag[copied * psf->DiagSize() + i] = b->diagnostics[i];
		}
		if (iterations) {
			for (int i = 0; i < b->numspots; i++)
				iterations[copied + i] = b->iterations[i];
		}
		if (likelihoods) {
			for (int i = 0; i < b->numspots; i++)
				likelihoods[copied + i] = b->ll[i];
		}
		copied += b->numspots;
		b->numspots = 0;
		LockedAction(recycleMutex, [&]() {
			recycle.push_back(std::move(b));
		});
	}
	return copied;
}

int LocalizationQueue::GetQueueLength()
{
	int sum = 0;
	LockedAction(todoMutex, [&]() {
		for (auto& b : todo)
			sum += b->numspots;
	});
	return sum;
}

void LocalizationQueue::Flush()
{
	if (next->numspots == 0)
		return;

	LockedAction(todoMutex, [&]() { todo.push_back(std::move(next)); });
	LockedAction(recycleMutex, [&]() {
		if (!recycle.empty()) {
			next = std::move(recycle.front());
			recycle.pop_front();
		}
	});
	if (!next) next = std::make_unique<Batch>(batchSize, psf);
}

void LocalizationQueue::Schedule(int count, const int* ids, const float * h_samples, const float * h_constants, const int * h_roipos)
{
	const float* smp = h_samples;
	const float* constants = h_constants;
	const int* roipos = h_roipos;

	for (int i = 0; i < count; i++) {
		Schedule(ids[i], smp, constants, roipos);
		smp += psf->SampleCount();
		constants += psf->NumConstants();
		roipos += psf->SampleIndexDims();
	}
}

void LocalizationQueue::Schedule(int id, const float * h_samples, const float * h_constants, const int * h_roipos)
{
	LockedAction(scheduleMutex, [&]() {
		int i = next->numspots++;

	//	DebugPrintf("Scheduling spot %d\n",id);

		next->ids[i] = id;
		for (int j = 0; j < numconst; j++)
			next->constants[i * numconst + j] = h_constants[j];
		for (int j = 0; j < smpcount; j++)
			next->samples[i * smpcount + j] = h_samples[j];
		for (int j = 0; j < smpdims; j++)
			next->roipos[i * smpdims + j] = h_roipos[j];

		if (next->numspots == batchSize) Flush();
	});
}


void LocalizationQueue::Launch(std::unique_ptr<Batch> b, LocalizationQueue::StreamData& sd)
{
	// Copy to GPU
	sd.samples.CopyToDevice(b->samples.data(), b->numspots*psf->SampleCount(), true, sd.stream);
	sd.constants.CopyToDevice(b->constants.data(), b->numspots*psf->NumConstants(), true, sd.stream);
	sd.roipos.CopyToDevice(b->roipos.data(), b->numspots*psf->SampleIndexDims(), true, sd.stream);

	// Process
	psf->Estimate(sd.samples.data(), sd.constants.data(), sd.roipos.data(), 0, sd.estimates.data(), 
		sd.diagnostics.data(), sd.iterations.data(), b->numspots, 0, 0, sd.stream);
	psf->FisherMatrix(sd.estimates.data(), sd.constants.data(), sd.roipos.data(), b->numspots, sd.fi.data(), sd.stream);
	//psf->FisherToCRLB(sd.fi.data(), sd.crlb.data(), b->numspots, sd.stream);

	// Copy results to host
	sd.estimates.CopyToHost(b->estimates.data(), b->numspots * psf->ThetaSize(), true, sd.stream);
	sd.diagnostics.CopyToHost(b->diagnostics.data(), b->numspots*psf->DiagSize(), true, sd.stream);
	sd.fi.CopyToHost(b->fisherInfo.data(), b->numspots*psf->ThetaSize()*psf->ThetaSize(), true, sd.stream);
	sd.crlb.CopyToHost(b->crlb.data(), b->numspots*psf->ThetaSize(), true, sd.stream);
	sd.iterations.CopyToHost(b->iterations.data(), b->numspots, true, sd.stream);

	sd.currentBatch = std::move(b);
}

LocalizationQueue::Batch::Batch(int batchsize, CUDA_PSF * psf)
	:numspots(0)
{
	ids.resize(batchsize);
	roipos.Init(batchsize*psf->SampleIndexDims());
	constants.Init(batchsize * psf->NumConstants());
	estimates.Init(batchsize * psf->ThetaSize());
	diagnostics.Init(batchsize*psf->DiagSize());
	samples.Init(batchsize * psf->SampleCount());
	fisherInfo.Init(batchsize*psf->ThetaSize()*psf->ThetaSize());
	crlb.Init(batchsize*psf->ThetaSize());
	iterations.Init(batchsize);
	ll.Init(batchsize);
}


CDLL_EXPORT LocalizationQueue* PSF_Queue_Create(PSF* psf, int batchSize, int maxQueueLen, int numStreams, Context* ctx)
{
	CUDA_PSF* cudapsf = psf->GetCUDA_PSF();
	if (!cudapsf)
		return 0;

	return new LocalizationQueue(cudapsf, batchSize, maxQueueLen, numStreams, ctx);
}

CDLL_EXPORT void PSF_Queue_Delete(LocalizationQueue* queue)
{
	delete queue;
}

CDLL_EXPORT void PSF_Queue_Schedule(LocalizationQueue* q, int numspots, const int* ids, const float* h_samples,
	const float* h_constants, const int* h_roipos)
{
	q->Schedule(numspots, ids, h_samples, h_constants, h_roipos);
}

CDLL_EXPORT void PSF_Queue_Flush(LocalizationQueue* q)
{
	q->Flush();
}

CDLL_EXPORT bool PSF_Queue_IsIdle(LocalizationQueue* q)
{
	return q->IsIdle();
}

CDLL_EXPORT int PSF_Queue_GetResultCount(LocalizationQueue* q)
{
	return q->GetResultCount();
}

CDLL_EXPORT int PSF_Queue_GetQueueLength(LocalizationQueue *q)
{
	return q->GetQueueLength();
}

// Returns the number of actual returned localizations. 
// Results are removed from the queue after they are copied to the provided memory
CDLL_EXPORT int PSF_Queue_GetResults(LocalizationQueue* q, int maxresults, float* estim, float* diag,
										int* iterations, float* ll, float *fi, float* crlb, int *roipos, int*ids)
{
	return q->GetResults(maxresults, estim, diag, iterations, ll, fi, crlb, roipos, ids);
}


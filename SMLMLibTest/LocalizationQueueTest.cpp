#include "PSFModels/PSF.h"
#include "PSFModels/Gaussian/GaussianPSF.h"
#include "LocalizationQueue.h"
#include "RandomDistributions.h"

void LocalizationQueueTest()
{
	float sigma = 1.6f;
	int roisize = 10;
	int numspots = 10;

	PSF* psf = Gauss2D_CreatePSF_XYIBg(roisize, sigma, true);
	auto queue = std::make_unique<LocalizationQueue>(psf->GetCUDA_PSF(), 256, 10, 3);

	std::vector<float> ev(psf->SampleCount());
	Vector4f theta{ roisize*0.5f,roisize*0.5f,100000.0f,2.0f };
	std::vector<int> roipos(psf->SampleIndexDims());

	psf->ExpectedValue(&ev[0],(const float*)&theta, 0, roipos.data(), 1);

	srand(time(0));
	for (int i = 0; i < 10; i++) {
		for (float& v : ev)
			v = (float)rand_poisson(v);

		queue->Schedule(i, &ev[0], 0, &roipos[0]);
	}
	queue->Flush();

	while (!queue->IsIdle()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		DebugPrintf(".");
	}
	int resultCount = queue->GetResultCount();
	std::vector<Vector4f> estimated(resultCount);
	std::vector<int> ids(resultCount);
	std::vector<Vector4f> crlb(resultCount);
	std::vector<float> fi(resultCount * 16);
	queue->GetResults(resultCount, (float*)estimated.data(), 0,0,0,fi.data(),(float*)crlb.data(),0,ids.data());

	Vector4f sum;
	for (int j = 0; j < resultCount; j++)
	{
		float* fi_=&fi[j*16];
		//InvertMatrix(4, fi_, P, tmpOut);


		auto v = estimated[j];
		DebugPrintf("id=%d, %f,%f,%f,%f\n", ids[j], v[0], v[1], v[2], v[3]);
		sum += estimated[j];
	}

	DebugPrintf("All done\n");

	queue.reset();
	delete psf;
}


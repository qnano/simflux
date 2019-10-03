#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <array>
#include <tuple>
#include <cassert>
#include <fstream>

#define NOMINMAX
#include <Windows.h>
#include <future>

#include "SolveMatrix.h"
#include "CudaUtils.h"
#include "MathUtils.h"
#include "PSFModels/Gaussian/GaussianPSF.h"
#include "Estimation.h"
#include "DebugImageCallback.h"
#include "RandomDistributions.h"
#include "PSFModels/BSpline/BSplinePSF.h"
#include "PSFModels/BSpline/BSplinePSFModels.h"
#include "PSFModels/Gaussian/GaussianPSFModels.h"
#include "simflux/SIMFLUX.h"
#include "TIFFReadWrite.h"
#include "GetPreciseTime.h"

#include "PSFModels/PSFImpl.h"

void TestLUInvert()
{
	float A[] = { 1,2,0,
		2,4,1,
		2,1,0 };

	float invertedShouldBe[] = {
		-1 / 3.0f,0,2.0f / 3,
		2.0f / 3,0,-1.0f / 3,
		-2,1,0 
	};

	DebugPrintf("Original:");
	printMatrix(A, 3, 3);

	float inverted[9];
	InvertMatrix<float,9>(A, inverted);

	DebugPrintf("Inverted:");
	printMatrix(inverted, 3, 3, "%.5f");

	for (int i=0;i<9;i++)
		if (fabs(inverted[i] - invertedShouldBe[i]) > 1e-7f)
		{
			DebugPrintf("Failed inversion");
			assert(0);
		}

	printMatrix(invertedShouldBe, 3, 3, "%.5f");


	const int K = 4;
	Vector<float,K*K> rnd;

	for (int i = 0; i < K*K; i++) {
		rnd[i] = rand_uniform<float>();
	}

	auto psd = MultiplyMatrix(rnd, TransposeMatrix(rnd));
	
	auto invPSD = InvertMatrix(psd);

	auto ident = MultiplyMatrix(psd, invPSD);

	printMatrix(ident.elem, K);
}

const float pi = 3.141592653589793f;

void HMMViterbiTest();
void ImageQueueTest();
void SIMFLUX_BlinkHMMTest();
void ZOLA3DTest();




void cudaInit();


void test(float&& x) {
	DebugPrintf("x: %f\n", x);
}

std::vector<float> sum_excitation_patterns(const std::vector<float>& src, int imgw, int numframes)
{
	std::vector<float> dst(src.size() / numframes);
	int n = (int)src.size() / (imgw*imgw*numframes);
	for (int i = 0; i<n; i++)
		for (int y = 0; y<imgw; y++)
			for (int x = 0; x < imgw; x++)
			{
				float sum = 0.0f;
				for (int e = 0; e < numframes; e++)
					sum += src[i*imgw*imgw*numframes + e * imgw*imgw + y * imgw + x];
				dst[i*imgw*imgw + y * imgw + x] = sum;
			}
	return dst;
}
std::vector<float> sum_silm_images(const std::vector<float>& src, int imgw, int numframes)
{
	int n = (int)src.size() / (imgw*imgw*numframes);
	std::vector<float> dst(n*numframes);
	for (int i = 0; i<n; i++) 
		for (int e = 0; e < numframes; e++) {
			float sum = 0.0f;
			for (int y = 0; y<imgw; y++)
				for (int x = 0; x < imgw; x++)
					sum += src[i*imgw*imgw*numframes + e * imgw*imgw + y * imgw + x];
			dst[i*numframes + e] = sum;
		}

	return dst;
}


template<typename T, int d>
void ApplyMatrix_(T* m, T* x, T* y) {
	for (int i = 0; i < d; i++)
	{
		T sum{};
		for (int j = 0; j < d; j++)
			sum += m[i*d + j] * x[j];
		y[i] = sum;
	}
}



void TestCholeskyInversion()
{
	float m[9] = {
		4,12,-16,
		12,37,43,
		-16,-43,98
	};
	float ch[9] = {};

	float W {};

	Cholesky(3, m, ch);

	float inv[9];
	float chInv[9] = {};

//https://scicomp.stackexchange.com/questions/3188/dealing-with-the-inverse-of-a-positive-definite-symmetric-covariance-matrix
//	for (int i = 0; i < 3; i++) {
	
	float x[3] = { 1,2,3};
	float y[3];
	ApplyMatrix<float,3>(m, x, y);

	float x_[3];
	ApplyMatrix<float,3>(inv, y, x_);

	DebugPrintf("Original: "); printMatrix(m, 3);
	DebugPrintf("Cholesky: "); printMatrix(ch, 3);
	DebugPrintf("Inverted: "); printMatrix(inv, 3);

	DebugPrintf("Original vector: "); PrintVector(x,3);
	DebugPrintf("Transformed: "); PrintVector(y,3);
	DebugPrintf("Transformed by inv: "); PrintVector(x_,3);
}

void debugImgCallback(int w, int h, int numImg, const float* data, const char *label)
{
	DebugPrintf("writing debug image %s (%d x %d)\n", label, w, h);

	if (numImg == 1) {
		WriteTIFF(SPrintf("%s.tiff", label).c_str(), data, w, h, false);
	}
	else {
		for (int i = 0; i < numImg; i++) {
			WriteTIFF(SPrintf("%s_%04d.tiff", label, i).c_str(), &data[i * w * h], w, h, false);
		}
	}
}

void TestDriftEstimation();

template<int DIM>
void test_coordinate_transform() {
	// Build simplex
	Cartesian<DIM> v1 = { {0., 0., 0.} };
	Cartesian<DIM> v2 = { {0., 0., 1.} };
	Cartesian<DIM> v3 = { {0., 1., 1.} };
	Cartesian<DIM> v4 = { {1., 1., 1.} };

	Simplex<DIM> ss;
	ss.vertices[0] = v1;
	ss.vertices[1] = v2;
	ss.vertices[2] = v3;
	ss.vertices[3] = v4;
	ss.min = v1;
	ss.max = v4;

	const Simplex<DIM> * s = &ss;
	double T[DIM][DIM] = {
	  s->vertices[0].x[0] - s->vertices[3].x[0], s->vertices[1].x[0] - s->vertices[3].x[0], s->vertices[2].x[0] - s->vertices[3].x[0],
	  s->vertices[0].x[1] - s->vertices[3].x[1], s->vertices[1].x[1] - s->vertices[3].x[1], s->vertices[2].x[1] - s->vertices[3].x[1],
	  s->vertices[0].x[2] - s->vertices[3].x[2], s->vertices[1].x[2] - s->vertices[3].x[2], s->vertices[2].x[2] - s->vertices[3].x[2]
	};

	// Prepare LU
	memcpy(ss.LU, T, DIM*DIM * sizeof(double));
	LUPDecompose<double, DIM>(ss.LU, 1e-9f, ss.pivot);

	Barycentric b;
	Cartesian p = { {1.0, 0.5, 0.5} };

	Spline::cart2bary(s, p, b);
	std::cout << b.x[0] << "\t" << b.x[1] << "\t" << b.x[2] << "\t" << b.x[3] << std::endl;
}

void LocalizationQueueTest();
void SimfluxTest();
void cudaBreak();
void SpotDetectTest();
void CheckCRLB();


int main() {
	SetDebugImageCallback(debugImgCallback);
	cudaInit();

	LocalizationQueueTest();

//	SpotDetectTest();
//	SimfluxTest();
//	TensorQueueTest();
//	ImageQueueTest();
//	LocalizationQueueTest();

	//test_coordinate_transform();
	return 0;

	//SplineTest();

	//TestLUInvert();
	//TestCholeskyInversion();
	//TestDriftEstimation();
	//HMMViterbiTest();

	//testGaussian2D();
	//testGaussian3D();
	//testCSpline();

	return 0;
}


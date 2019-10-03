#include "DriftEstimation.h"
#include "RandomDistributions.h"
#include "StringUtils.h"
#include "ContainerUtils.h"
#include "GetPreciseTime.h"

#include <Eigen/Dense>

std::vector<Vector3f> GenerateSpots(int w, int numspots)
{
	std::vector<Vector3f> spots(numspots);

	for (int i = 0; i < numspots; i++)
	{
		spots.push_back({ rand_uniform<float>()*w,rand_uniform<float>()*w, 100 });
	}
	return spots;
}

struct Test {
	int a = 1;
	int b = 2;
	const int c = 3;
};

void TestDriftEstimation()
{
	Test x;

	DebugPrintf("%d\n", sizeof(x));

	float driftSpeed = 0.5f; // drift speed per frame
	float locPrecision = 0.1f; // localization precision
	int w = 400, numframes=1000;
	float k_on=0.01f, k_off=0.1f;
	float p_on = k_on / (k_on + k_off);

	// Generate spots
	// Blink between frames
	// Add drift
	std::vector<Vector3f> spots = GenerateSpots(w, 100);

	std::vector<bool> blinkstate(spots.size());
	for (int i = 0; i < spots.size(); i++)
		blinkstate[i] = rand_uniform<float>() < p_on;

	std::vector<Vector3f> xyI;
	std::vector<int> indices;
	std::vector<int> oncount(numframes);
	Vector2f drift;
	std::vector<Vector2f> true_drift(numframes-1);
	for (int f = 0; f < numframes; f++) {
		for (int i = 0; i < spots.size(); i++) {
			if (blinkstate[i]) {
				Vector2f loc_error{ rand_normal<float>()*locPrecision, rand_normal<float>()*locPrecision };
				Vector2f pos = spots[i].xy() + drift + loc_error;
				xyI.push_back({ pos[0], pos[1], 100 });
				indices.push_back(f);
				blinkstate[i] = rand_uniform<float>() >= k_off;
			}
			else
				blinkstate[i] = rand_uniform<float>() < k_on;
		}
		oncount[f] = ArraySum(Cast<int>(blinkstate));
		drift[0] += (rand_uniform<float>() - 0.5f) * 2 * driftSpeed;
		drift[1] += (rand_uniform<float>() - 0.5f) * 2 * driftSpeed;
		if (f < numframes - 1) true_drift[f] = drift;
	}

	float searchDist = 4.0f;
	std::vector<Vector2f> drift_estim(numframes-1);
	std::vector<int> matchResults(numframes-1);

	std::vector<Int2> framePairs(numframes - 1);
	for (int i = 0; i < numframes - 1; i++) framePairs[i] = { i,i + 1 };

	Profile([&]() {
		NearestNeighborDriftEstimate(xyI.data(), indices.data(), xyI.size(), searchDist, framePairs.data(), framePairs.size(), 
			drift_estim.data(), matchResults.data(), w, w, 3);
	});

	Vector2f estim = {};
	for (int f = 0; f < numframes-1; f++) {
		estim += drift_estim[f];
		Vector2f err = true_drift[f] - estim;
		DebugPrintf("%f: true_drift={%4.3f,%4.3f} drift estim={%4.3f, %4.3f}. error: {%.3f,%.3f} #matching=%d / %d \n", f, 
			true_drift[f][0],true_drift[f][1], estim[0], estim[1], err[0], err[1], matchResults[f], oncount[f]);
	}
}


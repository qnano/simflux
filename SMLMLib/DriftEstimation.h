#pragma once

#include "DLLMacros.h"
#include "Vector.h"


CDLL_EXPORT void NearestNeighborDriftEstimate(const Vector3f * xyI, const int *spotFrameNum, int numspots, float searchDist, 
	const Int2 *framePairs, int numframePairs, Vector2f * drift, int *matchResults, int width, int height, int icpIterations);

CDLL_EXPORT void CrossCorrelationDriftEstimate(const Vector3f* xyI, const int *spotFrameNum, int numspots, const Int2* framePairs, int numframepairs,
	Vector2f* drift, int width, int height, float maxDrift);


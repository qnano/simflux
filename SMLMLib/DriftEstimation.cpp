#include "MemLeakDebug.h"
#include "palala.h"

#include "DriftEstimation.h"
#include "ContainerUtils.h"



std::vector<std::pair<int, int> > MatchSpots(const Vector3f* spotsA, int numA, const Vector3f* spotsB, int numB, Vector2f shiftEstim, float searchDist)
{
	std::vector<std::pair<int, int>> matching;
	std::vector<int> remainingB = Range(numB);

	// brute force matching... needs quadtree/grid
	const float searchDist2 = searchDist * searchDist;
	for (int i = 0; i < numA; i++)
	{
		Vector2f a = spotsA[i].xy() + shiftEstim;
		float bestDist2;
		int bestJ = -1;
		for (int j = 0; j < remainingB.size(); j++)
		{
			float dist2 = (a - spotsB[remainingB[j]].xy()).sqLength();
			if (dist2 < searchDist2 && (bestJ < 0 || bestDist2 > dist2))
			{
				bestJ = j;
				bestDist2 = dist2;
			}
		}

		if (bestJ >= 0) {
			matching.push_back({ i, remainingB[bestJ] });
			remainingB[bestJ] = remainingB.back();
			remainingB.pop_back();
		}
	}
	return matching;
}

struct Frame {
	int first, count;
};

std::vector<Frame> IndexFrames(const int * spotFrameNum, int numspots)
{
	int maxIndex = *std::max_element(spotFrameNum, spotFrameNum + numspots);
	int numFrames = maxIndex + 1;

	std::vector<Frame> frames(numFrames);
	int spot = 0, frame = 0;
	while (frame < numFrames) {
		frames[frame].first = spot;
		while (spotFrameNum[spot] == frame && spot < numspots) {
			frames[frame].count++;
			spot++;
		}
		frame++;
	}
	return frames;
}

CDLL_EXPORT void NearestNeighborDriftEstimate(const Vector3f * xyI, const int *spotFrameNum, int numspots, float searchDist,
	const Int2 *framePairs, int numFramePairs, Vector2f *drift, int *matchResults, int width, int height, int icpIterations)
{
	auto frames = IndexFrames(spotFrameNum, numspots);

	parallel_for_cpu(numFramePairs, [&](int p) { 
		Int2 pair = framePairs[p];
		const Vector3f* frame0 = xyI + frames[pair[0]].first;
		const Vector3f* frame1 = xyI + frames[pair[1]].first;

		Vector2f driftEstimate = drift[p];

		for (int i = 0; i < icpIterations; i++) {
			auto matching = MatchSpots(frame0, frames[pair[0]].count, frame1, frames[pair[1]].count, driftEstimate, searchDist);

			if (matching.size() > 0) {
				Vector2f diffSum = {};
				for (auto m : matching)
					diffSum += frame1[m.second].xy() - frame0[m.first].xy();

				driftEstimate = diffSum / (int)matching.size();
			}
			if (i == icpIterations - 1)
				matchResults[p] = matching.size();
		}

		drift[p] = driftEstimate;
	});
}


CDLL_EXPORT void CrossCorrelationDriftEstimate(const Vector3f* xyI, const int *spotFrameNum, int numspots, const Int2* framePairs, int numframepairs,
	Vector2f* drift, int width, int height, float maxDrift)
{

}



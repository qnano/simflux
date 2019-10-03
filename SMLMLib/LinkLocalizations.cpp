#include "DLLMacros.h"
#include "Vector.h"

CDLL_EXPORT int LinkLocalizations(int numspots, int* framenum, Vector3f* xyI, Vector3f* crlbXYI, float maxDist, float maxIntensityDist, int frameskip, 
									int *linkedSpots, int* startframes, int *framecounts, Vector3f* linkedXYI, Vector3f* linkedcrlbXYI)
{
	std::vector<Vector3f> linkedMeans (numspots), linkedVar(numspots);

	// Find number of framenum
	int nframes = 0;
	for (int i = 0; i < numspots; i++) {
		if (framenum[i] >= nframes) nframes = framenum[i] + 1;
	}

	// Organise by frame number
	std::vector<std::vector<int>> frameSpots (nframes);
	for (int i = 0; i < numspots; i++)
		frameSpots[framenum[i]].push_back(i);

	// Clear linked spots
	for (int i = 0; i < numspots; i++)
		linkedSpots[i] = -1;

	int nlinked = 0;
	auto linkspots = [&](int prev, int b) {
		// prev may or may not already be linked. b is definitely unlinked
		if (linkedSpots[prev] < 0) {
			linkedMeans[nlinked] = xyI[prev];
			linkedVar[nlinked] = crlbXYI[prev] * crlbXYI[prev];
			linkedSpots[prev] = nlinked++;
		}
		int j = linkedSpots[prev];
		linkedSpots[b] = j;

		Vector3f varB = crlbXYI[b]*crlbXYI[b];
		Vector3f totalVar = 1.0f / (1.0f / linkedVar[j] + 1.0f / varB);
		linkedMeans[j] = totalVar * (linkedMeans[j] / linkedVar[j] + xyI[b] / varB);
		linkedVar[j] = totalVar;
	};

	// Connect spots
	for (int f = 1; f < nframes; f++) {
		for (int b = std::max(0, f - frameskip-1); b < f; b++) {

			for (int i : frameSpots[f])
			{
				for (int prev : frameSpots[b]) {

					Vector3f cmpPos;
					if (linkedSpots[prev] >= 0)
						cmpPos = linkedMeans[linkedSpots[prev]];
					else
						cmpPos = xyI[prev];

					Vector2f diff = Vector2f{ xyI[i][0],xyI[i][1] } -Vector2f{ cmpPos[0],cmpPos[1] };
					float xydist = diff.length();
					float Idist = abs(xyI[i][2] - cmpPos[2]);

					// find spots within maxDist
					float maxPixelDist = maxDist;// *Vector2f{ crlbXYI[i][0],crlbXYI[i][1] }.length();
					if (xydist < maxPixelDist)///&& 
//						Idist < maxIntensityDist * crlbXYI[i][2])
					{
						linkspots(prev, i); // i should be linked to prev
						break;
					}
				}
			}

		}
	}

	// Give all non-linked spots a unique id
	for (int i = 0; i < numspots; i++)
	{
		if (linkedSpots[i] < 0)
			linkedSpots[i] = nlinked++;
	}

	std::vector<int> endframes(nlinked,-1);
	// Compute startframes/framecounts
	for (int i = 0; i < nlinked; i++) {
		startframes[i] = -1;
	}
	for (int i = 0; i < numspots; i++)
	{
		int j = linkedSpots[i];
		if (startframes[j]<0 || startframes[j]>framenum[i])
			startframes[j] = framenum[i];
		if (endframes[j] < 0 || endframes[j] < framenum[i])
			endframes[j] = framenum[i];
	}

	for (int i = 0; i < nlinked; i++)
		framecounts[i] = endframes[i] - startframes[i] + 1;

	// Compute spot mean xy and summed intensity
	for (int i = 0; i < nlinked; i++) {
		linkedXYI[i] = {};
		linkedXYI[i][0] = linkedMeans[i][0];
		linkedXYI[i][1] = linkedMeans[i][1];
		linkedcrlbXYI[i] = linkedVar[i].sqrt();
	}

	for (int i = 0; i < numspots; i++)
	{
		int j = linkedSpots[i];
		linkedXYI[j][2] += xyI[i][2];
	}

	return nlinked;
}



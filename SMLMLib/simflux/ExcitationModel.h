#pragma once

#include "palala.h"
#include <cuda_runtime.h>

struct SIMFLUX_Modulation;

class FixedFunctionExcitationPatternModel
{
public:
	const SIMFLUX_Modulation* modulation; // Each vector is [ kx, ky, phase ]
	int numep;

	PLL_DEVHOST int NumPatterns() const { return numep; }

	PLL_DEVHOST FixedFunctionExcitationPatternModel(int numep, const SIMFLUX_Modulation* modulation)
		: numep(numep), modulation(modulation) {}

	PLL_DEVHOST float BackgroundPattern(int e, int x, int y) const
	{
//		SIMFLUX_Modulation mod = modulation[e];
		// this seems to match reality the most (tiny variations in bg per pattern, not as much as mod.power)
		return 1.0f / numep; 
		//		return mod.power;
	}
	PLL_DEVHOST void ExcitationPattern(float& Q, float& dQdx, float& dQdy, int e, float2 xy) const {
		// compute Q, dQ/dx, dQ/dy
		SIMFLUX_Modulation mod = modulation[e];
		float A = mod.power;
		float w = mod.kx * xy.x + mod.ky * xy.y;
		float ang = w - mod.phase;
		Q = A * (1.0f + mod.depth * sin(ang));
		dQdx = A * mod.kx * mod.depth * cos(ang);
		dQdy = A * mod.ky * mod.depth * cos(ang);
	}
	PLL_DEVHOST float MeanBackgroundCoefficient() const
	{
		return 1.0f / numep;
	}
};

#pragma once

#include "DLLMacros.h"
#include "CameraCalibration.h"
#include "simflux/SIMFLUX.h"

class PSF;
struct SIMFLUX_Modulation;


CDLL_EXPORT PSF* SIMFLUX2D_PSF_Create(PSF* original, SIMFLUX_Modulation* mod, int num_patterns,
	const int * xyIBg_indices, bool simfluxFit, Context* ctx=0);

CDLL_EXPORT PSF* SIMFLUX2D_Gauss2D_PSF_Create(SIMFLUX_Modulation* mod, int num_patterns, float sigmaX, float sigmaY, 
	int roisize, int numframes, bool simfluxFit, bool defineStartEnd, sCMOS_Calibration* calib, Context* ctx=0);



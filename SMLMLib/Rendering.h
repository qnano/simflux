#pragma once

#include "DLLMacros.h"
#include "Vector.h"

CDLL_EXPORT void AddROIs(float* image, int width, int height, const float* rois, int numrois, int roisize, Int2* roiposYX);


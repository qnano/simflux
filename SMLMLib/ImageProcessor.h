#pragma once

#include "Context.h"

class ImageProcessor : public ContextObject {
public:
	virtual ~ImageProcessor() {}

	virtual void AddFrame(uint16_t* data) = 0;
	virtual int GetQueueLength() = 0;

	// Defaults to no output
	virtual int NumFinishedFrames() { return 0; }
	virtual int ReadFinishedFrame(float* original, float* processed) { return 0; }

	virtual bool IsIdle() = 0;
};


CDLL_EXPORT void ImgProc_AddFrame(ImageProcessor* q, uint16_t* data);
CDLL_EXPORT int ImgProc_GetQueueLength(ImageProcessor* p);
CDLL_EXPORT int ImgProc_ReadFrame(ImageProcessor* q, float* image, float* processed);
CDLL_EXPORT int ImgProc_NumFinishedFrames(ImageProcessor * q);
CDLL_EXPORT bool ImgProc_IsIdle(ImageProcessor* q);

class ROIExtractor;
struct ExtractionROI;
class IDeviceImageProcessor;

CDLL_EXPORT ROIExtractor* ROIExtractor_Create(int imgWidth, int imgHeight, ExtractionROI* rois,
	int numrois, int roiframes, int roisize, IDeviceImageProcessor* imgCalibration, Context* ctx);
CDLL_EXPORT int ROIExtractor_GetResultCount(ROIExtractor *re);
CDLL_EXPORT int ROIExtractor_GetResults(ROIExtractor* re, int numrois, ExtractionROI* rois, float* framedata);


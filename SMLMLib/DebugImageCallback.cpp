#include "MemLeakDebug.h"
#include "DebugImageCallback.h"
#include "CudaUtils.h"
#include <mutex>

std::mutex callbackMutex;
void(*debugImageCallback)(int w,int h, int numImages, const float* d, const char *title) = 0;


CDLL_EXPORT void SetDebugImageCallback(void(*cb)(int width, int height, int numImages, const float *data, const char *title))
{
	std::lock_guard<std::mutex> l(callbackMutex);
	debugImageCallback = cb;
}

CDLL_EXPORT void ShowDebugImage(int w, int h, int numImages, const float* data, const char *title)
{
	std::lock_guard<std::mutex> l(callbackMutex);

	if (debugImageCallback)
		debugImageCallback(w, h, numImages, data, title);
}

DLL_EXPORT void ShowDebugImage(const DeviceImage<float>& img, int numImages, const char *title)
{
	auto h_data = img.AsVector();
	ShowDebugImage(img.width, img.height/numImages, numImages, h_data.data(), title);
}



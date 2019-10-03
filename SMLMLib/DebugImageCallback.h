#pragma once
#include "DLLMacros.h"

CDLL_EXPORT void SetDebugImageCallback(void(*cb)(int width,int height, int numImg, const float* data, const char* title));
CDLL_EXPORT void ShowDebugImage(int w, int h, int numImg, const float* data, const char *title);

template<typename T>
class DeviceImage;

// Actual image height is img.h / numImg
DLL_EXPORT void ShowDebugImage(const DeviceImage<float>& img, int numImg, const char *title);
inline void ShowDebugImage(const DeviceImage<float>& img, const char *title) { ShowDebugImage(img, 1, title); }
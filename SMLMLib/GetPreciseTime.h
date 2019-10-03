#pragma once
#define NOMINMAX
#include <Windows.h>

inline double GetPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}

template<typename Fn>
void Profile(Fn cb) {
	double t0 = GetPreciseTime();
	cb();
	double t1 = GetPreciseTime();
	DebugPrintf("Elapsed time: %f s\n", t1 - t0);
}

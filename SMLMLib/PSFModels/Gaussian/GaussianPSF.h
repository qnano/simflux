#pragma once

#include "DLLMacros.h"
#include "Vector.h"
#include "Estimation.h"


class Context;
class PSF;
class sCMOS_Calibration;

typedef Vector4f Gauss2D_Theta;
typedef FisherMatrixType<Gauss2D_Theta>::type Gauss2D_FisherMatrix;
typedef EstimationResult<Gauss2D_Theta> Gauss2D_EstimationResult;

// spotList [ x y sigmaX sigmaY intensity ]
CDLL_EXPORT void Gauss2D_Draw(double* image, int imgw, int imgh, float* spotList, int nspots, float addSigma=0.0f);


class ISpotDetectorFactory;
struct ImageQueueConfig;



struct Gauss3D_Calibration
{
	Gauss3D_Calibration() {}
	float x[4];
	float y[4];
	float minz, maxz;
};

CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, sCMOS_Calibration *scmos = 0, Context* ctx=0);
CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda, sCMOS_Calibration *scmos = 0, Context* ctx=0);
CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda, sCMOS_Calibration *scmos = 0, Context* ctx=0);
CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, sCMOS_Calibration *sCMOS_calib, Context* ctx);

// Fit X,Y,intensity, and with a fixed background supplied as constant
CDLL_EXPORT PSF* Gauss2D_CreatePSF_XYI(int roisize, float sigma, bool cuda, sCMOS_Calibration *scmos = 0, Context* ctx=0);


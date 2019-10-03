#include "PSFModels/PSF.h"
#include "PSFModels/Gaussian/GaussianPSF.h"
#include "PSFModels/simflux/SIMFLUX_PSF.h"
#include "simflux/SIMFLUX_Models.h"
#include "simflux/ExcitationModel.h"
#include "TIFFReadWrite.h"
#include "RandomDistributions.h"

void SimfluxTest()
{
	const int roisize = 10;
	float sigma = 1.8f;
	PSF* g_psf = Gauss2D_CreatePSF_XYIBg(roisize, sigma, true);

	float pwr = 1.0f / 6;
	float depth = 0.95f;
	float kx = 1.8f;
	float ky = 1.8f;
	float pi = 3.141593f;

	std::vector<SIMFLUX_Modulation> mod = {
		{0, ky,   depth, 0, pwr},
		{kx, 0, depth, 0, pwr},
		{0, ky,   depth, 2 * pi / 3, pwr},
		{kx, 0, depth, 2 * pi / 3, pwr},
		{0, ky,  depth, 4 * pi / 3, pwr},
		{kx, 0, depth, 4 * pi / 3, pwr}
	};

	int xyIBg[] = { 0,1,2,3 };
//	PSF* sf_psf = SIMFLUX2D_PSF_Create(g_psf, mod.data(), mod.size(),xyIBg,false,0);
	PSF* sf_psf = SIMFLUX2D_Gauss2D_PSF_Create(mod.data(), mod.size(), sigma, roisize, 6, true, false, 0);

	int sf_roipos[3] = { 0,0,0 };
	std::vector<float> ev(roisize*roisize*mod.size());
	float theta[] = { roisize*0.5f+1,roisize*0.5f-0.3f, 1000, 2 };
	sf_psf->ExpectedValue(ev.data(), theta, 0, sf_roipos, 1);

	WriteTIFF("sf_psf.tiff", ev.data(), roisize, roisize*mod.size());

	Vector<float,16> sf_fi;
	sf_psf->FisherMatrix(theta, 0, sf_roipos, 1, sf_fi.elem);
	Vector4f sf_crlb = InvertMatrix(sf_fi).diagonal().sqrt();
	DebugPrintf("SF PSF CRLB: ");  PrintVector(sf_crlb);
	
	Vector<float, 16> g_fi;
	int g_roipos[2] = { 0,0 };
	g_psf->FisherMatrix(theta, 0, g_roipos, 1, g_fi.elem);
	auto g_crlb = InvertMatrix(g_fi).diagonal().sqrt();
	DebugPrintf("G2D CRLB: "); PrintVector(g_crlb);

	Vector<float, 16> sf_fi2;
	const SIMFLUX_ASW_Params p{ roisize,6,1.8f, 30,0.1f };
	std::vector<float > ev2(p.roisize*p.roisize*p.numepp);
	SIMFLUX_ASW_ComputeFisherMatrix(ev2.data(), &sf_fi2, mod.data(), (SIMFLUX_Theta*)theta, (const Int3*)sf_roipos, 1, 6, p);
	auto sf2_crlb = InvertMatrix(sf_fi2).diagonal().sqrt();
	DebugPrintf("SIMFLUX_ASW_ComputeFisherMatrix: "); PrintVector(sf2_crlb);

	FixedFunctionExcitationPatternModel epModel(p.numepp, mod.data());
	SIMFLUX_Calibration calib{ epModel, p.psfSigma };
	SIMFLUX_Model mdl(p.roisize, calib, 0, p.numepp, p.numepp);

	float q, dqdx, dqdy;
	for (int i = 0; i < mod.size(); i++) {
		epModel.ExcitationPattern(q, dqdx, dqdy, i, { theta[0],theta[1] });
		DebugPrintf("non-psf %d: ", i); PrintVector(Vector3f{ q,dqdx,dqdy });
	}

	std::vector<float> deriv(4 * roisize*roisize*mod.size()), ev3(roisize*roisize*mod.size());
	ComputeDerivatives(Vector4f(theta), mdl, Int3{}, SampleOffset_None<float>(), deriv.data(), ev3.data());
	auto sf3_crlb = ComputeCRLB(ComputeFisherMatrix(mdl, Int3{}, SampleOffset_None<float>(), Vector4f(theta), 0));
	DebugPrintf("SF Model-based CRLB: "); PrintVector(sf3_crlb);

	for (int i = 0; i < ev.size(); i++) {
//		assert(fabsf(ev3[i] - ev[i]) < 1e-5f);
	}

	std::vector<float> deriv2(4 * roisize*roisize*mod.size());
	sf_psf->Derivatives(deriv2.data(), ev.data(), theta, 0, sf_roipos, 1);

	for (int i = 0; i < deriv.size(); i++) {
	//	assert(fabsf(deriv2[i] -deriv[i])<1e-5f);
	}

	for (auto& v : ev)
		v = rand_poisson(v);

	WriteTIFF("simflux-psf.tif", &ev[0], roisize, roisize*mod.size());

	float estim[4] = {0};
	float diag[12];
	sf_psf->Estimate(ev.data(), 0, sf_roipos, 0, estim, diag, 0, 1, 0, 0);

	for (int i = 0; i < mod.size(); i++)
	{
		DebugPrintf("Pattern %d: I=%f, bg=%f\n", i, diag[i * 2 + 0], diag[i * 2 + 1]);
	}

	PrintVector(theta, 4);
	PrintVector(estim, 4);

//	printMatrix(ev.data(), roisize*mod.size(), roisize);
}

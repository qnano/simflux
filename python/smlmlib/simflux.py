# -*- coding: utf-8 -*-


import ctypes
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib as ctl
import scipy.stats
from enum import Enum

from .base import SMLM, NullableFloatArrayType
from smlmlib import gaussian
from smlmlib.context import Context
from smlmlib.psf import PSF
from smlmlib.calib import sCMOS_Calib

Theta = ctypes.c_float * 4
FisherMatrix = ctypes.c_float * 16
EstimationResult = gaussian.EstimationResult
Modulation = ctypes.c_float * 4




class SIMFLUX_ASW_Params(ctypes.Structure):
    _fields_ = [
        ("imgw", ctypes.c_int32),
        ("numep", ctypes.c_int32),
        ("sigma", ctypes.c_float),
        ("levMarMaxIt", ctypes.c_int32),
        ("levMarLambdaStep", ctypes.c_float)
    ]

    def make(imgw, numep, sigma, levMarIt=100, startLambdaStep=0.1):
        return SIMFLUX_ASW_Params(imgw, numep, sigma, levMarIt, startLambdaStep)


class SIMFLUX:
    def __init__(self, ctx:Context):
        self.ctx = ctx
        smlmlib = ctx.smlm.lib
        self._SIMFLUX_ASW_ComputeMLE = smlmlib.SIMFLUX_ASW_ComputeMLE
        self._SIMFLUX_ASW_ComputeMLE.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # img
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # modulation
            ctypes.POINTER(EstimationResult),  # results
            ctypes.c_int32,  # numspots
            ctypes.c_int32,  # numframes
            ctypes.POINTER(SIMFLUX_ASW_Params),  # p
            NullableFloatArrayType,  # initialValue
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
            ctypes.c_int32,  # flags
            NullableFloatArrayType,  # tracebuf
            ctypes.c_int32,  # tracebuflen per spot
        ]

        self._SIMFLUX_ASW_ComputeFisherMatrix = smlmlib.SIMFLUX_ASW_ComputeFisherMatrix
        self._SIMFLUX_ASW_ComputeFisherMatrix.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mu
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # fi
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # phi
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # theta
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # theta
            ctypes.c_int32,  # numspots
            ctypes.c_int32,  # numframes
            ctypes.POINTER(SIMFLUX_ASW_Params),
        ]
       # CDLL_EXPORT void SIMFLUX_DFT2D_Points(const Vector3f* xyI, int numpts, const Vector2f* k, 
       # int numk, Vector2f* output, bool useCuda);

        self._SIMFLUX_DFT2D_Points = smlmlib.SIMFLUX_DFT2D_Points
        self._SIMFLUX_DFT2D_Points.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctypes.c_int32,  # numpts
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # k
            ctypes.c_int32,  # numk
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # output
            ctypes.c_bool # useCuda
        ]

        # CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)
        self._FFT = smlmlib.FFT
        self._FFT.argtypes = [
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # src
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),  # dst
            ctypes.c_int32,  # batchsize
            ctypes.c_int32,  # numsigA
            ctypes.c_int32,  # forward
        ]
        
        self._SIMFLUX_ASW_ComputeOnOffProb = smlmlib.SIMFLUX_ASW_ComputeOnOffProb 
        self._SIMFLUX_ASW_ComputeOnOffProb.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # rois [numspots,numframes,roisize,roisize]
             ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mod[numep]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # gaussFits [numspots]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # IBg[out]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # probOnOff[numspots,numframes,2]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlbVariances[numspots,numframes,2]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # expectedIntensities[numspots,numframes]
            ctypes.POINTER(SIMFLUX_ASW_Params),  # p
            ctypes.c_int32,  # numframes
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # startPatterns[numspots]
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos[numspots]
            ctypes.c_bool,  # useCuda
                ]
        
        self._SIMFLUX_ProjectPointData = smlmlib.SIMFLUX_ProjectPointData
        self._SIMFLUX_ProjectPointData.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctypes.c_int32,  # numpts
            ctypes.c_int32,  # projectionWidth
            ctypes.c_float,  # scale
            ctypes.c_int32,  # numProjAngles
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # projectionAngles
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # output
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # output
        ]
        
#CDLL_EXPORT PSF* SIMFLUX2D_PSF_Create(PSF* original, SIMFLUX_Modulation* mod, int num_patterns, 
#	const int * xyIBg_indices)
             
        self._SIMFLUX2D_PSF_Create = smlmlib.SIMFLUX2D_PSF_Create
        self._SIMFLUX2D_PSF_Create.argtypes = [
                ctypes.c_void_p,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mod[numep]
                ctypes.c_int,
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  
                ctypes.c_bool,
                ctypes.c_void_p
            ]
        self._SIMFLUX2D_PSF_Create.restype = ctypes.c_void_p
        
#CDLL_EXPORT PSF* SIMFLUX2D_Gauss2D_PSF_Create(SIMFLUX_Modulation* mod, int num_patterns, 
# float sigma, int roisize, int numframes, bool simfluxFit, Context* ctx);
        
        self._SIMFLUX2D_Gauss2D_PSF_Create = smlmlib.SIMFLUX2D_Gauss2D_PSF_Create 
        self._SIMFLUX2D_Gauss2D_PSF_Create.argtypes= [
                
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mod[numep]
                ctypes.c_int, # numpatterns
                ctypes.c_float, # sigma_x
                ctypes.c_float, # sigma_y
                ctypes.c_int, # roisize
                ctypes.c_int, # nframes
                ctypes.c_bool, # simfluxfit
                ctypes.c_bool, # defineStartEnd
                ctypes.c_void_p, # scmos
                ctypes.c_void_p # context
                ]
        self._SIMFLUX2D_Gauss2D_PSF_Create.restype = ctypes.c_void_p
        
        #
#(int* spotToLinkedIdx, int *startframes, int *ontime,
#int numspots, int numlinked, int numpatterns, SpotToExtract* result)

        self._SIMFLUX_GenerateROIExtractionList = smlmlib.SIMFLUX_GenerateROIExtractionList
        self._SIMFLUX_GenerateROIExtractionList.argtypes= [
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # startframes
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # ontime
                ctypes.c_int, #maxresults
                ctypes.c_int, # numlinked
                ctypes.c_int, # numpatterns
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous")  # results
                ]

    def GenerateROIExtractionList(self, startframes, ontime, numpatterns):
        """
        returns linkedIndex, numroi and firstframe
        """
        maxresults = np.sum(ontime)//numpatterns
        numlinked = len(startframes)
        startframes=  np.ascontiguousarray(startframes,dtype=np.int32)
        ontime = np.ascontiguousarray(ontime, dtype=np.int32)
        results = np.zeros((maxresults,3),dtype=np.int32)
        resultcount = self._SIMFLUX_GenerateROIExtractionList(startframes,ontime,maxresults,numlinked,numpatterns,results)
        results =results[:resultcount]
        return results[:,0],results[:,1],results[:,2]

    
    def CreateSIMFLUX2DPSF(self, psf:PSF, mod, xyIBgIndices, simfluxEstim=False) -> PSF:
        xyIBgIndices=np.ascontiguousarray(xyIBgIndices,dtype=np.int32)
        assert(len(xyIBgIndices)==4)
        inst = self._SIMFLUX2D_PSF_Create(psf.inst, mod.astype(np.float32), len(mod), xyIBgIndices, 
                                          simfluxEstim, self.ctx.inst if self.ctx else None)
        return PSF(self.ctx,inst)
    
    class SIMFLUX_PSF(PSF):
        def __init__(self, ctx:Context, psfInst, mod):
            self.mod = mod
            super().__init__(self, ctx, psfInst)

        def ComputeExcitation(self,x,y):
            return self.mod[...,4]*(1+self.mod[...,2]*np.sin(self.mod[...,0]*x + self.mod[...,1]*y - self.mod[...,3]))

    def CreateSIMFLUX2D_Gauss2D_PSF(self, sigma, mod_or_num_patterns, roisize, 
                                    numframes, simfluxEstim=False, defineStartEnd=False, scmos_calib=None) -> PSF:
        if scmos_calib is not None:
            assert(isinstance(scmos_calib,sCMOS_Calib))
            scmos_calib = scmos_calib.inst

        mod = mod_or_num_patterns

        if mod is None:
            mod = 1
        if np.isscalar(mod):
            mod = np.zeros((mod,5))
        else:
            mod = np.ascontiguousarray(mod)
            assert(mod.shape[1] == 5)

        if np.isscalar(sigma):
            sigma_x, sigma_y = sigma,sigma
        else:
            sigma_x, sigma_y = sigma
            
        inst = self._SIMFLUX2D_Gauss2D_PSF_Create(mod.astype(np.float32), len(mod), sigma_x, sigma_y,
                                                  roisize, numframes, simfluxEstim, defineStartEnd,
                                                  scmos_calib, self.ctx.inst if self.ctx else None)
        return PSF(self.ctx,inst)

    # Convert an array of phases to an array of alternating XY modulation parameters
    def phase_to_mod(self, phases, omega, depth=1):
        mod = np.zeros((*phases.shape, 5), dtype=np.float32)
        mod[..., 0::2, 0] = omega  # kx
        mod[..., 1::2, 1] = omega  # ky
        mod[..., 2] = depth
        mod[..., 3] = phases
        mod[..., 4] = 1/len(mod)
        return mod

#CDLL_EXPORT void SIMFLUX_ASW_ComputeOnOffProb(const float* rois, 
#const SIMFLUX_Modulation* modulation, Vector4f* gaussFits, 
#	Vector2f* IBg, Vector2f* probOnOff, const SIMFLUX_ASW_Params& params, int numframes, 
#	int numspots, const int* startPatterns, const int2* roipos, bool useCuda)
    def SIMFLUX_ASW_ComputeOnOffProb(self, images, mod, xyIBg_gauss, silmParams: SIMFLUX_ASW_Params, 
                                  startPatterns, roipos, useCuda):
        mod = np.ascontiguousarray(mod, dtype=np.float32)
        images = np.ascontiguousarray(images, dtype=np.float32)
        xyIBg_gauss = np.ascontiguousarray(xyIBg_gauss,dtype=np.float32)
        numframes = images.shape[1]
        numspots = images.shape[0]
        
        probOnOff = np.zeros((numspots,numframes,2),dtype=np.float32)
        crlbVariances = np.zeros((numspots,numframes,2),dtype=np.float32)
        expectedIntensity = np.zeros((numspots,numframes),dtype=np.float32)
        IBg = np.zeros((numspots,numframes,2),dtype=np.float32)
        startPatterns = np.ascontiguousarray(startPatterns,dtype=np.int32)
        roipos = np.ascontiguousarray(roipos, dtype=np.int32)
        
        self._SIMFLUX_ASW_ComputeOnOffProb(
            images, mod, xyIBg_gauss, IBg, probOnOff, crlbVariances, expectedIntensity, silmParams, 
            numframes, numspots, startPatterns,roipos, useCuda)
        
        return probOnOff, IBg, crlbVariances, expectedIntensity
        

    def Params(self, imgw, numep, sigma, levMarIt=100, startLambdaStep=0.1):
        return SIMFLUX_ASW_Params(imgw, numep, sigma, levMarIt, startLambdaStep)
    
    
    def SIMFLUX_DFT2D_Points(self, xyI, k, useCuda=True):
        xyI = np.ascontiguousarray(xyI, dtype=np.float32)
        numpts = len(xyI)
        k = np.ascontiguousarray(k, dtype=np.float32)
        output = np.zeros( len(k), dtype=np.complex64)
        self._SIMFLUX_DFT2D_Points(xyI, numpts, k, len(k), output, useCuda)
        return output

    # CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth,
    # 	float scale, int numProjAngles, const float *projectionAngles, float* output)
    def ProjectPoints(self, xyI, projectionWidth, scale, projectionAngles):
        numProjAngles = len(projectionAngles)
        assert xyI.shape[1] == 3
        xyI = np.ascontiguousarray(xyI, dtype=np.float32)
        output = np.zeros((numProjAngles, projectionWidth), dtype=np.float32)
        shifts = np.zeros((numProjAngles), dtype=np.float32)

        self._SIMFLUX_ProjectPointData(
            xyI,
            len(xyI),
            projectionWidth,
            scale,
            numProjAngles,
            np.array(projectionAngles, dtype=np.float32),
            output,
            shifts,
        )
        return output, shifts

    ##CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)

    def FFT(self, src, forward=True):
        batchsize = len(src)
        src = np.ascontiguousarray(src, dtype=np.complex64)
        dst = np.zeros(src.shape, dtype=np.complex64)
        self._FFT(src, dst, batchsize, src.shape[1], forward)
        return dst

import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
from .base import NullableFloatArrayType
from smlmlib.context import Context

from smlmlib.calib import sCMOS_Calib

import scipy.stats

class PSF:
    def __init__(self, ctx:Context, psfInst):
        self.inst = psfInst
        self.ctx = ctx
        lib = ctx.smlm.lib
        self.copyROI = False

        InstancePtrType = ct.c_void_p

        self._PSF_Delete = lib.PSF_Delete
        self._PSF_Delete.argtypes = [InstancePtrType]

        self._PSF_ThetaSize = lib.PSF_ThetaSize
        self._PSF_ThetaSize.restype = ct.c_int32
        self._PSF_ThetaSize.argtypes = [InstancePtrType]

        self._PSF_ThetaFormat = lib.PSF_ThetaFormat
        self._PSF_ThetaFormat.restype = ct.c_char_p
        self._PSF_ThetaFormat.argtypes = [InstancePtrType]

        self._PSF_NumConstants = lib.PSF_NumConstants
        self._PSF_NumConstants.restype = ct.c_int32
        self._PSF_NumConstants.argtypes = [InstancePtrType]

        self._PSF_NumDiag = lib.PSF_NumDiag
        self._PSF_NumDiag.restype = ct.c_int32
        self._PSF_NumDiag.argtypes = [InstancePtrType]

        self._PSF_SampleSize = lib.PSF_SampleSize
        self._PSF_SampleSize.argtypes = [InstancePtrType, ct.c_int32]
        self._PSF_SampleSize.restype = ct.c_int32

        self._PSF_SampleIndexDims = lib.PSF_SampleIndexDims
        self._PSF_SampleIndexDims.restype = ct.c_int32
        self._PSF_SampleIndexDims.argtypes = [InstancePtrType]

        self._PSF_SampleCount = lib.PSF_SampleCount
        self._PSF_SampleCount.restype = ct.c_int32
        self._PSF_SampleCount.argtypes = [InstancePtrType]

        # CDLL_EXPORT void PSF_ComputeExpectedValue(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* ev);
        self._PSF_ComputeExpectedValue = lib.PSF_ComputeExpectedValue
        self._PSF_ComputeExpectedValue.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #theta 
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # constants
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # ev
        ]

        # CDLL_EXPORT void PSF_ComputeFisherMatrix(PSF* psf, int numspots, const float* theta, const float* constants,const int* spotpos,  float* fi);
        self._PSF_ComputeFisherMatrix = lib.PSF_ComputeFisherMatrix
        self._PSF_ComputeFisherMatrix.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #fi
        ]

        # CDLL_EXPORT void PSF_ComputeDerivatives(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* derivatives, float* ev);
        self._PSF_ComputeDerivatives = lib.PSF_ComputeDerivatives
        self._PSF_ComputeDerivatives.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
        ]
        
        self.fitInfoDType = np.dtype([('likelihood', '<f4'), ('iterations', '<i4')])

        # CDLL_EXPORT void PSF_ComputeMLE(PSF* psf, int numspots, const float* sample, const float* constants,const int* spotpos, 
        # const float* initial, float* theta, int* iterations, int maxiterations, float levmarAlpha, float* trace, int traceBufLen);
        self._PSF_ComputeMLE = lib.PSF_ComputeMLE
        self._PSF_ComputeMLE.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # sample
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # constants
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            NullableFloatArrayType,  # initial
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # theta
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # diagnostics
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # iterations
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # tracebuf [theta*numspots*tracebuflen]
            ct.c_int32,
        ]
        
        self.thetasize = self._PSF_ThetaSize(self.inst)
        self.numconst = self._PSF_NumConstants(self.inst)
        self.samplecount = self._PSF_SampleCount(self.inst)
        self.indexdims = self._PSF_SampleIndexDims(self.inst)
        self.samplesize = [0]*self.indexdims
        for i in range(self.indexdims):
            self.samplesize[i] = self._PSF_SampleSize(self.inst, i)
        self.numdiag = self._PSF_NumDiag(self.inst)
        self.fmt = self.ThetaFormat()
        self.colnames = self.fmt.split(',')

        #print(f"Created PSF Parameters: {self.fmt}, #const={self.numconst}. #diag={self.numdiag} samplesize={self.samplesize}." )

    def ThetaColNames(self):
        return self.colnames

    def _checkparams(self, theta, constants, roipos):
        theta = np.array(theta)
        if len(theta.shape) == 1:
            theta = [theta]
        theta = np.ascontiguousarray(theta, dtype=np.float32)
        numspots = len(theta)
        if constants is None:
            constants = np.zeros((numspots, self.numconst), dtype=np.float32)
        else:
            constants = np.ascontiguousarray(constants, dtype=np.float32)
            assert(np.array_equal(constants.shape, [numspots, self.numconst]))
        if roipos is None:
            roipos = np.zeros((numspots, self.indexdims), dtype=np.int32)
        else:
            roipos = np.ascontiguousarray(roipos, dtype=np.int32)
            assert(np.array_equal( roipos.shape, [numspots, self.indexdims]))

        if theta.shape[1] != self.thetasize:
            print(f"{self.fmt} expected, {theta.shape[1]} parameters given")
            assert(theta.shape[1]==self.thetasize)
        return theta,constants,roipos

    def ExpectedValue(self, theta, roipos=None, constants=None):
        theta,constants,roipos=self._checkparams(theta,constants,roipos)
        ev = np.zeros((len(theta), *self.samplesize), dtype=np.float32)
        self._PSF_ComputeExpectedValue(self.inst, len(theta), theta, constants, roipos, ev)
        return ev

    def GenerateSample(self, theta, roipos=None, constants=None):
        ev = self.ExpectedValue(theta, roipos, constants)
        return np.random.poisson(ev)

    def FisherMatrix(self, theta, roipos=None, constants=None):
        theta,constants,roipos=self._checkparams(theta,constants,roipos)
        fi = np.zeros((len(theta), self.thetasize, self.thetasize), dtype=np.float32)
        self._PSF_ComputeFisherMatrix(self.inst, len(theta), theta, constants, roipos, fi)
        return fi

    def CRLB(self, theta, roipos=None, constants=None):
        fisher = self.FisherMatrix(theta, roipos, constants)
        var = np.linalg.inv(fisher)
        crlb = np.sqrt(np.abs(var[:,np.arange(self.thetasize),np.arange(self.thetasize)]))
        return crlb
    
    def Derivatives(self, theta, roipos=None, constants=None):
        theta,constants,roipos=self._checkparams(theta,constants,roipos)
        deriv = np.zeros((len(theta), self.thetasize, *self.samplesize), dtype=np.float32)
        ev = np.zeros((len(theta), *self.samplesize), dtype=np.float32)
        self._PSF_ComputeDerivatives(self.inst, len(theta), theta, constants, roipos, deriv, ev)
        return deriv, ev
    
    def NumDeriv(self, theta, roipos=None, constants=None, eps=1e-4):
        theta,constants,roipos=self._checkparams(theta,constants,roipos)
        ev = self.ExpectedValue(theta,roipos,constants)
        deriv = np.zeros((len(theta), self.thetasize, *self.samplesize), dtype=np.float32)
        
        for k in range(self.thetasize):
            theta_min = theta*1
            theta_min[:,k] -= eps
            theta_max = theta*1
            theta_max[:,k] += eps
            ev_min = self.ExpectedValue(theta_min, roipos, constants)
            ev_max = self.ExpectedValue(theta_max, roipos, constants)
            deriv[:,k] = (ev_max-ev_min)/(2*eps)
        
        return deriv, ev
    
    def NumCRLB(self, theta, roipos=None, constants=None, eps=1e-4, useNumDeriv=True):
        if useNumDeriv:
            deriv,ev = self.NumDeriv(theta,roipos,constants,eps)
        else:
            deriv,ev = self.Derivatives(theta,roipos,constants)

        K = self.thetasize
        deriv = np.reshape(deriv, (len(theta), K, self.samplecount))
        ev[ev<1e-9] = 1e-9
        fi = np.zeros((len(theta), K,K))
        for i in range(K):
            for j in range(K):
                fi[:,i,j] = np.sum(1/ev * (deriv[:,i] * deriv[:,j]), axis=1)
        var = np.linalg.inv(fi)
        return np.sqrt(np.abs(var[:,np.arange(self.thetasize),np.arange(self.thetasize)]))

    def ComputeMLE(self, imgs, roipos=None, constants=None, initial=None, maxiterations=50, levmarInitialAlpha=0.1):
        numspots = len(imgs)
        
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        assert np.array_equal(imgs.shape, [numspots, *self.samplesize])
        traceBufLen = maxiterations
        iterations = np.zeros(numspots, np.int32)
        theta = np.zeros((numspots, self.thetasize), dtype=np.float32)
        trace = np.zeros((numspots, traceBufLen, self.thetasize), dtype=np.float32)
        diag = np.zeros((numspots, self.numdiag), dtype=np.float32)

        theta,constants,roipos = self._checkparams(theta,constants,roipos)

        if initial is not None:
            initial = np.ascontiguousarray(initial, dtype=np.float32)
            assert len(initial) == numspots

        if numspots>0:
            self._PSF_ComputeMLE(
                self.inst,
                numspots,
                imgs,
                constants,
                roipos,
                initial,
                theta,
                diag,
                iterations,
                trace,
                traceBufLen
            )

        traces = []
        for k in range(numspots):
            traces.append(trace[k, 0 : iterations[k] + 1, :])

        return theta, diag, traces

    def Destroy(self):
        if self.inst is not None:
            self._PSF_Delete(self.inst)
            self.inst = None

    def NumConstants(self):
        return self.numconst

    def ThetaFormat(self):
        return self._PSF_ThetaFormat(self.inst).decode("utf-8") 
    
    def FindThetaIndex(self, paramName):
        return self.ThetaFormat().split(',').index(paramName)

    def ThetaSize(self):
        return self.thetasize

    def SampleCount(self):
        return self.thetasize
    
    def NumDiag(self):
        return self.numdiag
    
    def Draw(self, image, thetas, roiposYX, constants=None):
        rois = self.ExpectedValue(thetas, roiposYX, constants)
        return self.ctx.smlm.DrawROIs(image, rois, roiposYX)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()


    @staticmethod
    def CenterOfMassEstimator(roisize, ctx: Context=None):
        smlmlib = ctx.smlm.lib

        fn = smlmlib.CreateCenterOfMassEstimator
        fn.argtypes = [
                ct.c_int32,
                ct.c_void_p]
        fn.restype = ct.c_void_p

        inst = fn(roisize, ctx.inst)
        return PSF(ctx, inst)



    @staticmethod
    def PhasorEstimator(roisize, ctx: Context=None):
        smlmlib = ctx.smlm.lib

        fn = smlmlib.CreatePhasorEstimator
        fn.argtypes = [
                ct.c_int32,
                ct.c_void_p]
        fn.restype = ct.c_void_p

        inst = fn(roisize, ctx.inst)
        return PSF(ctx, inst)

    @staticmethod 
    def CopyROI_Create(psf, ctx:Context):
        _CopyROI_CreatePSF = ctx.smlm.lib.CopyROI_CreatePSF
        _CopyROI_CreatePSF.argtypes = [
                ct.c_void_p,
                ct.c_void_p]
        _CopyROI_CreatePSF.restype = ct.c_void_p

        inst = _CopyROI_CreatePSF(psf.inst, ctx.inst if ctx else None)
        psf = PSF(ctx, inst)
        psf.copyROI = True
        return psf
    
    @staticmethod 
    def GLRT_PSF_Create(psf, ctx:Context, scmos : sCMOS_Calib=None):
#CDLL_EXPORT PSF * GLRT_CreatePSF(PSF* model, Context* ctx, sCMOS_Calibration* calib)
        _GLRT_CreatePSF = ctx.smlm.lib.GLRT_CreatePSF
        _GLRT_CreatePSF.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p
                ]
        _GLRT_CreatePSF.restype = ct.c_void_p

        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst

        inst = _GLRT_CreatePSF(psf.inst, ctx.inst if ctx else None, 
                               scmos)
        psf = PSF(ctx, inst)
        return psf
    


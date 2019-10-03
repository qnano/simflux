import ctypes
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib as ctl
import numbers

from smlmlib.context import Context
from smlmlib import psf
from smlmlib import spotdetect
from .base import SMLM, NullableFloatArrayType
from smlmlib.calib import sCMOS_Calib

Theta = ctypes.c_float * 4
FisherMatrix = ctypes.c_float * 16

class EstimationResult(ctypes.Structure):
    _fields_ = [
        ("estimate", Theta),
        ("initialValue", Theta),
        ("crlb", Theta),
        ("iterations", ctypes.c_int32),
        ("roiPosition", ctypes.c_int32 * 2),
        ("score", ctypes.c_float),
        ("loglikelihood", ctypes.c_float),
    ]


class Gauss3D_Calibration(ctypes.Structure):
    _fields_ = [
            ("x", ctypes.c_float * 4), 
            ("y", ctypes.c_float * 4),
            ("zrange", ctypes.c_float*2)
        ]

    def __init__(self, x, y, zrange=[-0.3, 0.3]):
        self.x = (ctypes.c_float * 4)(*x)
        self.y = (ctypes.c_float * 4)(*y)
        self.zrange = (ctypes.c_float*2)(*zrange)

    @classmethod
    def from_file(cls, filename):
        calibration = np.load(filename, allow_pickle=True).item()
        return cls(calibration.get("x"), calibration.get("y"))


class Gaussian:
    def __init__(self, ctx: Context):
        smlmlib = ctx.smlm.lib
        self.ctx = ctx
        self.lib = ctx.smlm

        # CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, int numspots, const Vector2f* xy,
        # const float *sigma, int imgw, int maxIterations, bool cuda)
        self._EstimateIBg = smlmlib.Gauss2D_EstimateIntensityBg
        self._EstimateIBg.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # images
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # IBg (result)
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # IBg_crlb (result)
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xy (input)
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # sigma
            ctypes.c_int32,  # imgw
            ctypes.c_int32,  # maxiterations
            ctypes.c_int32  # cuda
        ]
        
        self._Gauss2D_IntensityBg_CRLB = smlmlib.Gauss2D_IntensityBg_CRLB
        self._Gauss2D_IntensityBg_CRLB.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlb
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyIBg
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
            ctypes.c_float,  # sigma
            ctypes.c_int32,  # imgw
            ctypes.c_bool  # cuda
        ]
         
        # (float * image, int imgw, int imgh, float * spotList, int nspots)
        self._Gauss2D_Draw = smlmlib.Gauss2D_Draw
        self._Gauss2D_Draw.argtypes = [
            ctl.ndpointer(np.float64, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32,
            ctypes.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32,
            ctypes.c_float
        ]

        self._Gauss2D_CreatePSF_XYIBg = smlmlib.Gauss2D_CreatePSF_XYIBg
        self._Gauss2D_CreatePSF_XYIBg.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float,  # sigmax
                ctypes.c_float,  # sigmay
                ctypes.c_int32,  # cuda
                ctypes.c_void_p, # scmos calib
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBg.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYIBgSigma = smlmlib.Gauss2D_CreatePSF_XYIBgSigma
        self._Gauss2D_CreatePSF_XYIBgSigma.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float, 
                ctypes.c_int32,  # cuda
                ctypes.c_void_p, # scmos calib
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBgSigma.restype = ctypes.c_void_p
        
        self._Gauss2D_CreatePSF_XYIBgSigmaXY = smlmlib.Gauss2D_CreatePSF_XYIBgSigmaXY
        self._Gauss2D_CreatePSF_XYIBgSigmaXY.argtypes = [
                ctypes.c_int32,  # roisize
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_int32,  # cuda
                ctypes.c_void_p, # scmos calib
                ctypes.c_void_p] # context
        self._Gauss2D_CreatePSF_XYIBgSigmaXY.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYI = smlmlib.Gauss2D_CreatePSF_XYI
        self._Gauss2D_CreatePSF_XYI.argtypes = [
                ctypes.c_int32, 
                ctypes.c_float, 
                ctypes.c_int32, 
                ctypes.c_void_p,
                ctypes.c_void_p]
        self._Gauss2D_CreatePSF_XYI.restype = ctypes.c_void_p

        self._Gauss2D_CreatePSF_XYZIBg = smlmlib.Gauss2D_CreatePSF_XYZIBg
        self._Gauss2D_CreatePSF_XYZIBg.argtypes = [
                ctypes.c_int32, 
                ctypes.POINTER(Gauss3D_Calibration), 
                ctypes.c_int32, 
                ctypes.c_void_p,
                ctypes.c_void_p]
        self._Gauss2D_CreatePSF_XYZIBg.restype = ctypes.c_void_p
        

    def ComputeFisherMatrix(self, theta, sigma, imgw):
        with self.CreatePSF_XYIBg(imgw, sigma, False) as psf:
            fi = psf.FisherMatrix(theta)
            mu = psf.ExpectedValue(theta)
            return mu, fi

    # Spots is an array with rows: [ x,y, sigmaX, sigmaY, intensity ]
    def Draw(self, img, spots, addSigma=0):
        spots = np.ascontiguousarray(spots, dtype=np.float32)
        nspots = spots.shape[0]
        assert spots.shape[1] == 5
        img = np.ascontiguousarray(img, dtype=np.float64)
        self._Gauss2D_Draw(img, img.shape[1], img.shape[0], spots, nspots, addSigma)
        return img

    def Plot(self, theta, sigma, imgw, sample=False):
        mu, fi = self.ComputeFisherMatrix(theta, sigma, imgw)
        mu = mu[0]
        if sample:
            mu = np.random.poisson(mu)

        plt.figure()
        plt.imshow(mu)

    def EstimateIBg(self, images, sigma, xy, roipos=None, useCuda=False):
        images = np.ascontiguousarray(images, dtype=np.float32)
        xy = np.ascontiguousarray(xy, dtype=np.float32)
        assert len(images.shape) == 3
        numspots = len(images)
        assert np.array_equal( xy.shape, (numspots, 2))
        imgw = images.shape[1]
        result = np.zeros((numspots, 2), dtype=np.float32)
        crlb = np.zeros((numspots, 2), dtype=np.float32)
        
        if type(sigma) != np.ndarray:
            sigma = np.ones(len(xy),dtype=np.float32)*sigma
        else:
            assert np.array_equal(sigma.shape,(numspots,))
            sigma = np.ascontiguousarray(sigma)
        
        if roipos is None:
            roipos = np.zeros((numspots,2))
        assert np.array_equal(roipos.shape, (numspots,2))
        roipos = np.ascontiguousarray(roipos,dtype=np.int32)
        if numspots > 0:
            self._EstimateIBg(images, result, crlb, numspots, xy, roipos, sigma, imgw, 100, useCuda)
        return result, crlb

#CDLL_EXPORT void Gauss2D_IntensityBg_CRLB(Vector2f* IBg_crlb,
#	int numspots, const Vector4f* xyIBg, float sigma, int imgw, bool cuda)
    def CRLB_IBg(self, xy, IBg, roipos, roisize, psfSigma, useCuda=False):
        numspots = len(xy)
        assert xy.shape == (numspots, 2)
        assert IBg.shape == (numspots, 2)
        xyIBg = np.zeros((numspots,4),dtype=np.float32)
        xyIBg[:,[0,1]] = xy
        xyIBg[:,[2,3]] = IBg
        crlb = np.zeros((numspots, 2), dtype=np.float32)
        roipos = np.ascontiguousarray(roipos,dtype=np.int32)
        if numspots > 0:
            self._Gauss2D_IntensityBg_CRLB(crlb, numspots, xyIBg, roipos, psfSigma, roisize, useCuda)
        return crlb

    # CDLL_EXPORT PSF* Gauss_CreatePSF_XYIBg(int roisize, float sigma, bool cuda);
    def CreatePSF_XYIBg(self, roisize, sigma, cuda, scmos=None) -> psf.PSF:
        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst
        if np.isscalar(sigma):
            sigma_x, sigma_y = sigma,sigma
        else:
            sigma_x, sigma_y = sigma

        inst = self._Gauss2D_CreatePSF_XYIBg(roisize, sigma_x, sigma_y, cuda, scmos, self.ctx.inst)
        return psf.PSF(self.ctx, inst)
    
    def CreatePSF_XYIBgSigma(self, roisize, initialSigma, cuda, scmos=None) -> psf.PSF:
        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst
        inst = self._Gauss2D_CreatePSF_XYIBgSigma(roisize, initialSigma, cuda, scmos, self.ctx.inst)
        return psf.PSF(self.ctx, inst)

    def CreatePSF_XYIBgSigmaXY(self, roisize, initialSigma, cuda, scmos=None) -> psf.PSF:
        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst
        inst = self._Gauss2D_CreatePSF_XYIBgSigmaXY(roisize, initialSigma[0], initialSigma[1], cuda, scmos, self.ctx.inst)
        return psf.PSF(self.ctx, inst)

    def CreatePSF_XYI(self, roisize, sigma, cuda, scmos=None) -> psf.PSF:
        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst
        inst = self._Gauss2D_CreatePSF_XYI(roisize, sigma, cuda, scmos, self.ctx.inst)
        return psf.PSF(self.ctx, inst)

    # CDLL_EXPORT PSF* Gauss_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda);
    def CreatePSF_XYZIBg(self, roisize, calib: Gauss3D_Calibration, cuda, scmos=None) -> psf.PSF:
        if scmos is not None:
            assert(isinstance(scmos,sCMOS_Calib))
            scmos = scmos.inst
        inst = self._Gauss2D_CreatePSF_XYZIBg(roisize, calib, cuda, scmos, self.ctx.inst)
        return psf.PSF(self.ctx, inst)


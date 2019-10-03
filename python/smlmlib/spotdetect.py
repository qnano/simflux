# -*- coding: utf-8 -*-

import ctypes
from .base import SMLM
import numpy as np
import numpy.ctypeslib as ctl
import numbers

from smlmlib.image_proc import ImageProcessor
from smlmlib.psf_queue import PSF_Queue

class SpotDetectorNativeFactory:
    def __init__(self, inst, destructor):
        self.inst = inst
        self.destructor = destructor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        d = self.destructor
        d(self.inst)


class SpotDetectorConfig(ctypes.Structure):
    # detectionImage = uniform1 - uniform2
    # Selected spot locations = max(detectionImage, maxFilterSize) == detectionImage
    _fields_ = [
        ("uniformFilter1Size", ctypes.c_int32),
        ("uniformFilter2Size", ctypes.c_int32),
        ("maxFilterSize", ctypes.c_int32),
        ("roisize", ctypes.c_int32),  # Roisize is used to remove ROIs near the image border
        ("minIntensity", ctypes.c_float),  # Only spots where detectionImage > intensityThreshold are selected
        ("maxIntensity", ctypes.c_float),
    ]  # Only spots where detectionImage > intensityThreshold are selected

    def __init__(self, psfSigma, roisize, minIntensity=10, maxIntensity=np.inf):
        psfSigma = np.mean(psfSigma)
        self.uniformFilter1Size = int(psfSigma * 2 + 2)
        self.uniformFilter2Size = self.uniformFilter1Size * 2
        self.maxFilterSize = int(psfSigma * 5)
        self.roisize = roisize
        self.minIntensity = minIntensity
        self.maxIntensity = maxIntensity
        
        self.print()

    def print(self):
        print("Spot detector config: \n" +
              f"uniform filter 1: {self.uniformFilter1Size}\n"
              f"uniform filter 2: {self.uniformFilter2Size}\n"
              f"maxfilter: {self.maxFilterSize}\n"
              )

    def CreateNativeFactory(self, ctx):
        m = SpotDetectionMethods(ctx)
        return SpotDetectorNativeFactory(m._SpotDetector_Configure(self), m._SpotDetector_DestroyFactory)


class GLRTSpotDetectorConfig:
    sigma = 2
    maxspots = 500
    roisize = 10
    fdr = 0.5
    def __init__(self, sigma, roisize, fdr, maxspots=500):
        self.sigma = sigma
        self.roisize = roisize
        self.fdr = fdr
        self.maxspots = maxspots

    def CreateNativeFactory(self, ctx):
        m = SpotDetectionMethods(ctx)
        return SpotDetectorNativeFactory(
                m._GLRT_Configure(self.sigma,self.maxspots, self.fdr, self.roisize),
                m._SpotDetector_DestroyFactory)


class SpotDetectionMethods:
    def __init__(self, ctx):
        InstancePtrType = ctypes.c_void_p
        self.ctx = ctx
        self.lib = ctx.smlm
        lib = ctx.smlm.lib

        self._SpotDetector_Configure = lib.SpotDetector_Configure
        self._SpotDetector_Configure.argtypes = [ctypes.POINTER(SpotDetectorConfig)]
        self._SpotDetector_Configure.restype = ctypes.c_void_p

        self._SpotDetector_DestroyFactory = lib.SpotDetector_DestroyFactory
        self._SpotDetector_DestroyFactory.argtypes = [InstancePtrType]


#CDLL_EXPORT ISpotDetectorFactory* GRLT_Configure(float psfSigma, int maxSpots, float fdr, int roisize);

        self._GLRT_Configure = lib.GLRT_Configure
        self._GLRT_Configure.argtypes = [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int]
        self._GLRT_Configure.restype = ctypes.c_void_p

#CDLL_EXPORT int SpotDetector_ProcessFrame(const float* frame, int width, int height,
#	int maxSpots, float* spotScores, Int2* cornerPos, float* rois, const SpotDetectorConfig & cfg)

        self._SpotDetector_ProcessFrame = lib.SpotDetector_ProcessFrame
        self._SpotDetector_ProcessFrame.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # frame
            ctypes.c_int32,  # width
            ctypes.c_int32,  # height
            ctypes.c_int32,  # roisize
            ctypes.c_int32,  # maxspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # spotscores
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # cornerpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # rois [output]
            ctypes.c_void_p,
            ctypes.c_void_p, # Calibration object
            ]
        self._SpotDetector_ProcessFrame.restype = ctypes.c_int32

        #CDLL_EXPORT void ExtractROIs(const float *frames, int width, int height, int depth,
        # int roiX, int roiY, int roiZ, const Int3 * startpos, int numspots, float * rois);
        self._ExtractROIs = lib.ExtractROIs
        self._ExtractROIs.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # frames
            ctypes.c_int32,  # width
            ctypes.c_int32,  # height
            ctypes.c_int32,  # depth
            ctypes.c_int32,  # roisizeX
            ctypes.c_int32,  # roisizeY
            ctypes.c_int32,  # roisizeZ
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # startZYX
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # rois
        ]
        
#CDLL_EXPORT SpotLocalizerQueue * SpotLocalizerQueue_Create(int w,int h,, LocalizationQueue* queue, 
#	ISpotDetectorFactory* spotDetectorFactory, IDeviceImageProcessor* preprocessor, 
#	int numDetectionThreads, Context* ctx)
        
        self._SpotLocalizerQueue_Create = lib.SpotLocalizerQueue_Create
        self._SpotLocalizerQueue_Create.argtypes =[
            ctypes.c_int32,  # w
            ctypes.c_int32,  # h
            ctypes.c_void_p, # psf queue
            ctypes.c_void_p, # spotDetectorFactory
            ctypes.c_void_p, # calib
            ctypes.c_int32,  # nthreads
            ctypes.c_int32,  # sumframes
            ctypes.c_void_p,  #context
                ]
        self._SpotLocalizerQueue_Create.restype = ctypes.c_void_p
        
    def CreateLocalizationQueue(self, imgshape, psf_queue: PSF_Queue, spotDetectorConfig:SpotDetectorConfig,
                                calib=None, sumframes=1, numThreads=3, ctx=None) -> ImageProcessor:
        
        if ctx is None:
            ctx = self.ctx

        with spotDetectorConfig.CreateNativeFactory(self.ctx) as sdf:
            inst = self._SpotLocalizerQueue_Create(imgshape[1],imgshape[0],
                                            psf_queue.inst, 
                                            sdf.inst, 
                                            calib.inst if calib else None, 
                                            numThreads, 
                                            sumframes,
                                            ctx.inst if ctx else None)

        return ImageProcessor(imgshape, inst, self.ctx)

    def ExtractROIs(self, frames, roisize, cornerPosZYX):
        assert(len(frames.shape)==3)
        assert(cornerPosZYX.shape[1] == 3)

        numspots = len(cornerPosZYX)
        cornerPosZYX = np.ascontiguousarray(cornerPosZYX,dtype=np.int32)
        frames = np.ascontiguousarray(frames,dtype=np.float32)
        rois = np.zeros((numspots, *roisize),dtype=np.float32)

        self._ExtractROIs(frames, frames.shape[2], frames.shape[1], frames.shape[0],
                          roisize[2],roisize[1],roisize[0],cornerPosZYX,numspots,rois)

        return rois


    def ProcessFrame(self, image, spotDetectorConfig:SpotDetectorConfig, maxSpotsPerFrame, calib=None):
        assert len(image.shape)==2
        h = image.shape[0]
        w = image.shape[1]

        image = np.ascontiguousarray(image,dtype=np.float32)

        scores = np.zeros(maxSpotsPerFrame, dtype=np.float32)
        roisize=spotDetectorConfig.roisize
        rois = np.zeros((maxSpotsPerFrame, roisize,roisize),dtype=np.float32)
        cornerYX = np.zeros((maxSpotsPerFrame, 2),dtype=np.int32)

        with spotDetectorConfig.CreateNativeFactory(self.ctx) as sdf:
            numspots = self._SpotDetector_ProcessFrame(image, w,h,roisize,maxSpotsPerFrame,
                                                       scores,cornerYX,rois,sdf.inst,
                                                       calib.inst if calib else None)

        rois = rois[:numspots]
        scores = scores[:numspots]
        cornerYX = cornerYX[:numspots]

        return rois, cornerYX, scores

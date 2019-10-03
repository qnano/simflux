# -*- coding: utf-8 -*-
import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
from smlmlib.context import Context
from smlmlib.image_proc import ImageProcessor


class BlinkDetectorQueueConfig(ct.Structure):
    _fields_ = [
    	("width", ct.c_int32),
    	("height", ct.c_int32),
    	("removeBgFilterSizes", ct.c_int32*2),
        ("minmaxPeriod", ct.c_int32),
        ("minmaxSizeXY", ct.c_int32),
        ("roisize",  ct.c_int32),
        ("turnOnThreshold", ct.c_float),
        ("turnOffThreshold", ct.c_float),
        ("maxROIframes", ct.c_int32),
        ("maxPeakDist", ct.c_float)
    ]

    @staticmethod
    def make(imgshape, uniformFilter1=5,uniformFilter2=12, roisize=10,
                 maxFilterT=6, maxFilterXY=6, startThreshold=10, endThreshold=10, maxROIframes=100, maxPeakDist=5):
                
        return BlinkDetectorQueueConfig(imgshape[1],imgshape[0],(ct.c_int32*2)(uniformFilter1,uniformFilter2),maxFilterT,
                                        maxFilterXY,roisize, startThreshold, endThreshold, maxROIframes, maxPeakDist)


class BlinkFilterQueue(ImageProcessor):
    def __init__(self, cfg:BlinkDetectorQueueConfig, timefilter, calib, ctx:Context):
        self.frameCounter = 0
        lib = ctx.lib
        InstancePtrType = ct.c_void_p
        self.imgshape = (cfg.height, cfg.width)
        self.cfg = cfg
        
#BlinkDetectorConfig& cfg, IDeviceImageProcessor* calib, 
#	const float* timefilter, Context* ctx)
        createfn = ctx.lib.BlinkDetectorQueue_Create
        createfn.argtypes = [
            ct.POINTER(BlinkDetectorQueueConfig),
            ct.c_void_p, #calib
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #timefilter
            ct.c_int32, # timefilterlen
            ct.c_void_p, #ctx
                ]
        createfn.restype = InstancePtrType
        
        #CDLL_EXPORT int BlinkDetector_GetSpotCount(BlinkDetectorQueue* q)
        self._BlinkDetector_GetSpotCount = ctx.lib.BlinkDetector_GetSpotCount
        self._BlinkDetector_GetSpotCount.argtypes = [
                ct.c_void_p
                ]
        self._BlinkDetector_GetSpotCount.restype= ct.c_int32

        """
        struct BlinkDetectorROI
        {
        	int startFrame, numFrames;
        	float startPeak, endPeak;
        	Int2 cornerXY;
        };
        """

        self.blinkROIDtype = np.dtype([
                ("startframe", np.int32),
                ("numframes", np.int32),
                ("onpeak", np.float32),
                ("offpeak", np.float32),
                ("cx", np.int32),
                ("cy", np.int32)
                ])

#void BlinkDetector_GetSpots(BlinkDetectorQueue*q, int count, float* data, BlinkDetectorROI* rois)
        self._BlinkDetector_GetSpots = ctx.lib.BlinkDetector_GetSpots
        self._BlinkDetector_GetSpots.argtypes = [
                ct.c_void_p,
                ct.c_int32,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #data
                ctl.ndpointer(self.blinkROIDtype, flags="aligned, c_contiguous"), #timefilter
                ]
        
        timefilter = np.ascontiguousarray(timefilter,dtype=np.float32)
        inst = createfn(cfg, calib.inst if calib else None, timefilter, len(timefilter), ctx.inst if ctx else None)
        
        super().__init__(self.imgshape,inst,ctx)

    def GetSpotCount(self):
        return self._BlinkDetector_GetSpotCount(self.inst)

    def GetSpots(self, maxSpots=-1):
        c = self.GetSpotCount()
        if maxSpots>=0: c = np.minimum(maxSpots,c)
        spots = np.empty(c, dtype=self.blinkROIDtype)
        data = np.zeros((c, self.cfg.maxROIframes, self.cfg.roisize, self.cfg.roisize),dtype=np.float32)
        self._BlinkDetector_GetSpots(self.inst, c, data, spots)
        
        return spots, data

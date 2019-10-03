# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import ctypes
from .base import SMLM
import numpy as np
import numpy.ctypeslib as ctl


class PostProcess:
    def __init__(self, smlmlib):
        self.lib = smlmlib

        # CDLL_EXPORT void NearestNeighborDriftEstimate(const Vector3f * xyI, const int *spotFrameNum,
        # int numspots, float searchDist, const Int2 *framePairs, int numFramePairs, Vector2f *drift,
        # int *matchResults, int width, int height, int icpIterations)

        self._NearestNeighborDriftEstimate = smlmlib.lib.NearestNeighborDriftEstimate
        self._NearestNeighborDriftEstimate.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI Vector3f[]
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # indices int[]
            ctypes.c_int32,  # numspots
            ctypes.c_float,  # searchDist
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framepairs int2[]
            ctypes.c_int32,  # numframepairs
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # drift Vector2f[numframepairs]
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # match results int[numframepairs]
            ctypes.c_int32,  # width
            ctypes.c_int32,  # height
            ctypes.c_int32,  # icpIterations
        ]
        #CDLL_EXPORT void LinkLocalizations(int numspots, int* frames, Vector2f* xyI, float maxDist, int frameskip, int *linkedSpots)

        self._LinkLocalizations = smlmlib.lib.LinkLocalizations
        self._LinkLocalizations.argtypes = [
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framenum
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlbXYI
            ctypes.c_float,  # maxdist (in crlbs)
            ctypes.c_float, # max intensity distance (in crlb's)
            ctypes.c_int32,  # frameskip
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # linkedspots
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # startframes
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framecounts
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # linkedXYI
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # linkedCRLBXYI
        ]
        self._LinkLocalizations.restype = ctypes.c_int32
    
    def LinkLocalizations(self, xyI, crlbXYI, framenum, maxdist, maxIntensityDist, frameskip):
        """
        linked: int [numspots], all spots that are linked will have the same index in linked array.
        """
        xyI = np.ascontiguousarray(xyI,dtype=np.float32)
        crlbXYI = np.ascontiguousarray(crlbXYI,dtype=np.float32)
        framenum = np.ascontiguousarray(framenum, dtype=np.int32)
        linked = np.zeros(len(xyI),dtype=np.int32)
        framecounts = np.zeros(len(xyI),dtype=np.int32)
        startframes = np.zeros(len(xyI),dtype=np.int32)
        resultXYI = np.zeros(xyI.shape,dtype=np.float32)
        resultCRLBXYI = np.zeros(crlbXYI.shape,dtype=np.float32)
        
        assert crlbXYI.shape[1] == 3
        assert xyI.shape[1] == 3
        assert len(xyI) == len(crlbXYI)
        
        nlinked = self._LinkLocalizations(len(xyI), framenum, xyI, crlbXYI, maxdist, maxIntensityDist, 
                                          frameskip, linked, startframes, framecounts, resultXYI, resultCRLBXYI)
        startframes = startframes[:nlinked]
        framecounts = framecounts[:nlinked]
        resultXYI = resultXYI[:nlinked]
        resultCRLBXYI = resultCRLBXYI[:nlinked]
        return linked, framecounts,startframes, resultXYI, resultCRLBXYI

    def NearestNeighborDriftEstimate(self, xyI, indices, framepairs, imgshape, searchDist=2, icpIterations=4):
        xyI = np.ascontiguousarray(xyI, dtype=np.float32)
        numspots = xyI.shape[0]

        if xyI.shape[1] == 2:
            xyI_ = np.zeros((len(xyI), 3), dtype=np.float32)
            xyI_[:, 0:2] = xyI
            xyI = xyI_

        assert xyI.shape[1] == 3
        assert framepairs.shape[1] == 2

        indices = np.ascontiguousarray(indices, dtype=np.int32)

        framepairs = np.ascontiguousarray(framepairs, dtype=np.int32)
        numFramePairs = len(framepairs)
        drift = np.zeros((numFramePairs, 2), dtype=np.float32)
        matchResults = np.zeros((numFramePairs), dtype=np.int32)

        self._NearestNeighborDriftEstimate(
            xyI,
            indices,
            numspots,
            searchDist,
            framepairs,
            numFramePairs,
            drift,
            matchResults,
            imgshape[1],
            imgshape[0],
            icpIterations,
        )

        return drift, matchResults

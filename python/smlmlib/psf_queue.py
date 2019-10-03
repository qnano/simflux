# -*- coding: utf-8 -*-


import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
import time
import copy

from smlmlib.psf import PSF
from smlmlib.base import NullableFloatArrayType, NullableIntArrayType

class PSF_Queue_Results:
    def __init__(self, colnames, sampleshape, estim, diag, fi, crlb, iterations, ll, roipos, ids):
        self.estim = estim
        self.sampleshape = sampleshape
        self.diagnostics = diag
        self.fisher = fi
        self.crlb = crlb
        self.roipos = roipos
        self.iterations = iterations
        self.ll = ll
        self.ids = ids
        self.colnames = colnames

    def CRLB(self):
        K = self.estim.shape[1]
        var = np.linalg.inv(self.fisher)
        return np.sqrt(np.abs(var[:,np.arange(K),np.arange(K)]))
    
    def SortByID(self, isUnique=False):
        if isUnique:
            order = np.arange(len(self.ids))
            order[self.ids] = order*1
        else:
            order = np.argsort(self.ids)
        self.Filter(order)
        return order       
        
    def Filter(self, indices):
        if indices.dtype == bool:
            indices = np.nonzero(indices)[0]
        
        if len(indices) != len(self.ids):
            print(f"Removing {len(self.ids)-len(indices)}/{len(self.ids)}")
        self.estim = self.estim[indices]
        self.diagnostics = self.diagnostics[indices]
        self.fisher = self.fisher[indices]
        self.roipos = self.roipos[indices]
        self.iterations = self.iterations[indices]
        self.ll = self.ll[indices]
        self.ids = self.ids[indices]
        
#        if self.crlb is not None:
 #           self.crlb = self.crlb[indices]
        return indices
        
    def FilterXY(self, minX, minY, maxX, maxY):
        return self.Filter(np.where(
            np.logical_and(
                np.logical_and(self.estim[:,0]>minX, self.estim[:,1]>minY),
                np.logical_and(self.estim[:,0]<maxX, self.estim[:,1]<maxY)))[0])

    def Clone(self):
        return copy.deepcopy(self)
    
    def AsROIs(self):
        return self.diagnostics.reshape((len(self.estim), *self.sampleshape))
    
    def ColIdx(self, *names):
        return np.squeeze(np.array([self.colnames.index(n) for n in names],dtype=np.int))
        
    @staticmethod
    def Merge(result_list):
        estim=[] 
        diag=[]
        fi=[]
        roipos=[]
        it=[]
        ids=[]
        ll=[]
        for r in result_list:
            estim.append(r.estim)
            diag.append(r.diag)
            fi.append(r.fi)
            roipos.append(r.roipos)
            it.append(r.it)
            ids.append(r.ids)
            ll.append(r.ll)
            
        return PSF_Queue_Results(result_list[0].colnames, result_list[0].sampleshape,
                 estim=np.concatenate(estim),
                 diag=np.concatenate(diag),
                 fi=np.concatenate(fi),
                 iterations=np.concatenate(it),
                 ll=np.concatenate(ll),
                 roipos=np.concatenate(roipos),
                 ids=np.concatenate(ids))
        
    def SaveHDF5(self,fn, imgshape):
        from utils.picasso_hdf5 import save as save_hdf5
        
        idx = self.ColIdx('x','y','I','bg')
        crlb = self.CRLB()
        
        if 'sigma' in self.colnames:
            sx=sy=self.estim[:,self.colnames.index('sigma')]
        elif 'sx' in self.colnames:
            sx=self.estim[:,self.colnames.index('sx')]
            sy=self.estim[:,self.colnames.index('sy')]
        else:
            sx=sy=np.median(crlb[:,self.ColIdx('x','y')])
            
        xyIBg = self.estim[:,idx] * 1
        xyIBg[:,[0,1]] += self.roipos[:,[-1,-2]]
        
        #def save(fn, xyIBg, crlb, framenum, imgshape, sigmaX, sigmaY, extraColumns=None):   
        save_hdf5(fn, xyIBg, crlb[:,idx], self.ids, imgshape, sx, sy)

class PSF_Queue:
    def __init__(self, psf:PSF, batchSize=256, maxQueueLenInBatches=5, numStreams=-1,ctx=None):
        if ctx is None:
            self.ctx = psf.ctx
        else:
            self.ctx = ctx
        lib = self.ctx.smlm.lib
        self.psf = psf
        self.batchSize = batchSize

        InstancePtrType = ct.c_void_p

#        DLL_EXPORT LocalizationQueue* PSF_CreateQueue(PSF* psf, int batchSize, int maxQueueLen, int numStreams);
        self._PSF_Queue_Create = lib.PSF_Queue_Create
        self._PSF_Queue_Create.argtypes = [
                InstancePtrType, 
                ct.c_int32,
                ct.c_int32,
                ct.c_int32,
                ct.c_void_p]
        self._PSF_Queue_Create.restype = InstancePtrType
        
#        DLL_EXPORT void PSF_DeleteQueue(LocalizationQueue* queue);
        self._PSF_Queue_Delete= lib.PSF_Queue_Delete
        self._PSF_Queue_Delete.argtypes = [InstancePtrType]
        
#        DLL_EXPORT void PSF_Queue_Schedule(LocalizationQueue* q, int numspots, const int *ids, const float* h_samples,
 #       	const float* h_constants, const int* h_roipos);

        self._PSF_Queue_Schedule = lib.PSF_Queue_Schedule
        self._PSF_Queue_Schedule.argtypes = [
                InstancePtrType,
                ct.c_int32,
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # ids
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # samples
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # const
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
                ]
        
#        DLL_EXPORT void PSF_Queue_Flush(LocalizationQueue* q);
        self._PSF_Queue_Flush = lib.PSF_Queue_Flush
        self._PSF_Queue_Flush.argtypes = [InstancePtrType]
        
#        DLL_EXPORT bool PSF_Queue_IsIdle(LocalizationQueue* q);
        self._PSF_Queue_IsIdle = lib.PSF_Queue_IsIdle
        self._PSF_Queue_IsIdle.argtypes = [InstancePtrType]
        self._PSF_Queue_IsIdle.restype = ct.c_bool
        
#        DLL_EXPORT int PSF_Queue_GetResultCount(LocalizationQueue* q);
        self._PSF_Queue_GetResultCount = lib.PSF_Queue_GetResultCount
        self._PSF_Queue_GetResultCount.argtypes = [InstancePtrType]
        self._PSF_Queue_GetResultCount.restype = ct.c_int32

        self._PSF_Queue_GetQueueLength = lib.PSF_Queue_GetQueueLength
        self._PSF_Queue_GetQueueLength.argtypes = [InstancePtrType]
        self._PSF_Queue_GetQueueLength.restype = ct.c_int32
        
#        // Returns the number of actual returned localizations. 
 #       // Results are removed from the queue after copying to the provided memory
#        DLL_EXPORT int PSF_Queue_GetResults(LocalizationQueue* q, int maxresults, float* estim, float* diag, float *fi);
        self._PSF_Queue_GetResults = lib.PSF_Queue_GetResults
        self._PSF_Queue_GetResults.argtypes = [
                InstancePtrType,
                ct.c_int32,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # estim
                NullableFloatArrayType, # diag
                NullableIntArrayType, # iterations
                NullableFloatArrayType, # ll
                NullableFloatArrayType, # fi
                NullableFloatArrayType, # crlb
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # roipos
                NullableIntArrayType, # ids
                ]
        self._PSF_Queue_GetResults.restype = ct.c_int32
        
        self.colnames = psf.ThetaColNames()
        
        self.inst = self._PSF_Queue_Create(psf.inst,batchSize, maxQueueLenInBatches, numStreams, 
                                           self.ctx.inst if self.ctx else None)
        if not self.inst:
            raise RuntimeError("Unable to create PSF MLE Queue with given PSF")
            
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()

    def Destroy(self):
        self._PSF_Queue_Delete(self.inst)
        
    def Flush(self):
        self._PSF_Queue_Flush(self.inst)
        
    def WaitUntilDone(self):
        self.Flush()
        while not self.IsIdle():
            time.sleep(0.05)

        
    def IsIdle(self):
        return self._PSF_Queue_IsIdle(self.inst)
    
    def Schedule(self, samples, roipos=None, ids=None, constants=None):
        samples = np.ascontiguousarray(samples,dtype=np.float32)
        numspots = len(samples)

        if roipos is None:
            roipos = np.zeros((numspots,self.psf.indexdims),dtype=np.int32)
            
        if constants is None:
            constants = np.zeros((numspots, self.psf.numconst), dtype=np.int32)

        constants = np.ascontiguousarray(constants,dtype=np.float32)
        roipos = np.ascontiguousarray(roipos, dtype=np.int32)
        if (self.psf.numconst>0):
            assert(np.array_equal(constants.shape, [numspots, self.psf.numconst]))
        assert(np.array_equal(roipos.shape, [numspots, self.psf.indexdims]))
        assert(np.array_equal(self.psf.samplesize, samples.shape[1:]))
        
        if ids is None:
            ids = np.zeros(numspots,dtype=np.int32)
        else:
            assert len(ids) == len(samples)
            ids = np.ascontiguousarray(ids,dtype=np.int32)
        
        self._PSF_Queue_Schedule(self.inst, numspots, ids, samples, constants, roipos)
        
    def GetQueueLength(self):
        return self._PSF_Queue_GetQueueLength(self.inst)
    
    def GetResultCount(self):
        return self._PSF_Queue_GetResultCount(self.inst)
        
    def GetResults(self,maxResults=None) -> PSF_Queue_Results:
        count = self._PSF_Queue_GetResultCount(self.inst)
        
        if maxResults is not None and count>maxResults:
            count=maxResults
        
        K = self.psf.ThetaSize()
        estim = np.zeros((count, K),dtype=np.float32)
        diag = np.zeros((count, self.psf.NumDiag()), dtype=np.float32)
        fi = np.zeros((count, K,K), dtype=np.float32)
        crlb = np.zeros((count, K), dtype=np.float32)
        iterations = np.zeros(count, dtype=np.int32)
        roipos = np.zeros((count, self.psf.indexdims),dtype=np.int32)
        ll = np.zeros((count), dtype=np.float32)

        ids = np.zeros(count, dtype=np.int32)
        copied = self._PSF_Queue_GetResults(self.inst, count, estim, diag, iterations, ll, fi, crlb, roipos, ids)
        assert(count == copied)
        
        r = PSF_Queue_Results(self.colnames, self.psf.samplesize, 
                                 estim, diag, fi, crlb, iterations, ll, roipos, ids)
                    
        return r
        



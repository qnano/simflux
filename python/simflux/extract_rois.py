
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import sys
    sys.path.append('..')


from smlmlib.context import Context
from smlmlib.base import SMLM
from smlmlib.psf import PSF
from smlmlib.psf_queue import PSF_Queue, PSF_Queue_Results
import smlmlib.spotdetect as spotdetect
from smlmlib.simflux import SIMFLUX
from smlmlib.postprocess import PostProcess
from smlmlib.image_proc import ROIExtractor
from utils.picasso_hdf5 import load as load_hdf5

from smlmlib.util import imshow_hstack
import time

import simflux.read_tiff as read_tiff
from simflux.tiff_to_locs import create_calib_obj

class ROIData:
    def __init__(self, rois, frames, imgshape, estim, crlb, iterations, ibg, ibg_crlb):
        self.rois=rois
        self.frames=frames
        self.imgshape=imgshape
        self.estim=estim
        self.crlb=crlb
        self.iterations=iterations
        self.ibg=ibg
        self.ibg_crlb=ibg_crlb

    @staticmethod
    def from_results(rois,frames,imgshape, lr: PSF_Queue_Results):
        assert(len(lr.ids)==len(rois))
        d = lr.diagnostics.reshape((len(rois),frames.shape[1],4))
        ibg = d[:,:,0:2]
        ibg_crlb = d[:,:,2:4]
        return ROIData(rois, frames, imgshape, lr.estim,lr.CRLB(),lr.iterations,ibg,ibg_crlb) 

    def save(self,rois_path):
        with open(rois_path,"wb") as f:
            np.save(f, self.rois)
            np.save(f, self.imgshape)
            np.save(f, self.frames)
            np.save(f, self.estim)
            np.save(f, self.crlb)
            np.save(f, self.iterations)
            np.save(f, self.ibg)
            np.save(f, self.ibg_crlb)
        
    @staticmethod
    def load(rois_path):
        with open(rois_path,"rb") as f:
            rois = np.load(f)
            imgshape = np.load(f)
            frames = np.load(f)
            estim = np.load(f)
            crlb = np.load(f)
            iterations = np.load(f)
            ibg = np.load(f)
            ibg_crlb = np.load(f)
        return ROIData(rois,frames,imgshape,estim,crlb,iterations,ibg,ibg_crlb)

def localize(rois,frames,sigma,mod=None) -> PSF_Queue_Results:
    
    startend = np.zeros((len(rois),2),dtype=np.float32)
    startend[:,0] = 0
    startend[:,1] = rois['numframes']
    
    simfluxEstim = (mod is not None)
    
    roipos = np.zeros((len(frames),3),dtype=np.int32)
    roipos[:,0] = rois['startframe']
    roipos[:,[2,1]] = rois['cornerpos']

    with SMLM() as smlm, Context(smlm) as ctx:
        numframes = frames.shape[1]
        roisize = frames.shape[2]
        psf = SIMFLUX(ctx).CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, numframes, simfluxEstim, defineStartEnd=True)
        
        with PSF_Queue(psf) as q:
            q.Schedule(frames,roipos,constants=startend,ids=np.arange(len(rois)))
            q.WaitUntilDone()
            r = q.GetResults()
            print(f"Filtering {len(r.estim)} spots...")
            minX = 2.1
            minY = 2.1
            r.FilterXY(minX,minY,roisize-minX-1, roisize-minY-1)
#            plt.figure()
 #           plt.hist(r.iterations)
            r.Filter(np.where(r.iterations<80)[0])
            return r
    
def roll_per_row(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
    
    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]
    
    return A[rows, column_indices]    
    
    
def ibg_per_pattern(rois_path, num_patterns, sigma):
    roidata = ROIData.load(rois_path)
    
    # ignore last 6 ibg
    ibg = roidata.ibg[:,:num_patterns,:]

    shifts = roidata.rois['startframe'] % num_patterns
    ibg = roll_per_row(ibg, shifts)
    
    return ibg
 
def create_roi_extraction_list(locs_fn, roisize, maxdist, maxIntensityDist, frameskip, minroiframes,maxroiframes, ctx):
    # Load localizations and build a roi extraction list
    estim,framenum,crlb,imgshape = load_hdf5(locs_fn)
    print(f"{locs_fn} contains {len(estim)} spots")

    api = PostProcess(ctx.smlm)
    linked, ontime, startframes, resultXYI, resultCRLBXY = api.LinkLocalizations(
            estim[:,:3],crlb[:,:3],framenum,maxdist,maxIntensityDist,frameskip)

    roipos = (resultXYI[:,:2]+0.5).astype(int) - roisize//2
    sel = np.nonzero(ontime > minroiframes)[0]
    
    roilist = np.zeros(len(sel),dtype=ROIExtractor.ROIType)
    roilist['cornerpos'] = roipos[sel]
    roilist['startframe'] = startframes[sel]
    roilist['numframes'] = np.minimum(ontime[sel], maxroiframes)
    
    print(f"Spot extraction list contains {len(sel)} localizations (from {len(ontime)} linked spots.), {np.sum(roilist['numframes'])} ROIs.")
    return roilist
    
def extract_rois(rois_path, tiff_path, cfg, minroiframes, maxroiframes, appendframes, locs_fn, progress_update):
    """
    Extract all ROIs
    2D Gaussian fit on sums, and estimate intensity + backgrounds on individual frames
    """
    roisize = cfg['roisize']
    gain = cfg['gain']
    offset = cfg['offset']
    startframe = cfg['startframe'] if 'startframe' in cfg else 0
    maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
    
    maxdistXY = cfg['maxlinkdistXY']
    maxdistI = cfg['maxlinkdistI']
    frameskip = cfg['maxlinkframeskip']
    
    with SMLM(debugMode=False) as smlm, Context(smlm) as ctx:

        if progress_update is not None:
            progress_update("Generating extraction list...",0)

        extractionList = create_roi_extraction_list(locs_fn, roisize, maxdistXY, maxdistI, 
                                                    frameskip, minroiframes, maxroiframes, ctx)
        
        memsize = maxroiframes * len(extractionList) * 4*roisize*roisize / 1024
        print(f"ROIs size in memory: {memsize} KiB" )
   
        tiff_iterator = read_tiff.tiff_read_file(tiff_path, startframe, maxframes, progress_update)
    
        imgshape = read_tiff.tiff_get_image_size(tiff_path)
        calib = create_calib_obj(gain,offset,imgshape,ctx)
                
        with Context(smlm) as iq_ctx:
            q = ROIExtractor(imgshape, extractionList, maxroiframes, roisize, calib, iq_ctx)

            numframes = 0
            for fr,img in tiff_iterator:
                q.PushFrame(img)
                numframes += 1

            while not q.IsIdle():
                time.sleep(0.1)

            # Did user abort?                
            if progress_update is not None:
                if not progress_update("Localizing and estimating per-frame intensities",None): return None,None

            resultcount = q.GetResultCount()
            print(f"Result count: {resultcount}. Saving to {rois_path}")

            rois,frames = q.GetResults(resultcount)

        r = localize(rois,frames,cfg['sigma'])
        r.SaveHDF5(rois_path+".hdf5",imgshape)
        spot_rois = ROIData.from_results(rois[r.ids],frames[r.ids],imgshape,r)
        spot_rois.save(rois_path)
        return rois,frames

        
if __name__ == "__main__":

    #base='C:/dev/simflux/data/7-23-2019/Pos3/1_merge'
    base='C:/data/gattabeads/gattabeads-2_ld_1'
    rois_path=base+'.rois'
    tiff_path=base+'.tif'
    locs_fn=base+'.hdf5'
    
    minroiframes = 5
    maxroiframes = 60
    append_frames = 0
    
    cfg={
        'sigma': 1.83,
        'roisize': 9,
        'gain': 2.2,
        'offset': 100,
        'startframe': 0,
#        'maxframes': 100,
        'maxlinkdistXY': 0.3,
        'maxlinkdistI': 3,
        'maxlinkframeskip': 1
        }    
    rois,frames=extract_rois(rois_path, tiff_path, cfg, minroiframes, maxroiframes, append_frames, locs_fn, None)
    
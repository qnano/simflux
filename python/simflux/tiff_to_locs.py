

if __name__ == "__main__":
    import sys
    sys.path.append('..')

import numpy as np
import time
import tqdm
import os

from smlmlib.gaussian import Gaussian
from smlmlib.base import SMLM
import smlmlib.spotdetect as spotdetect
from smlmlib.calib import GainOffset_Calib
from smlmlib.calib import GainOffsetImage_Calib
from smlmlib.context import Context
from smlmlib.psf_queue import PSF_Queue


import simflux.read_tiff as read_tiff

def process_movie(imgshape, sdcfg, calib, psf_queue:PSF_Queue, movie, sumframes):
    t0 = time.time()
    
    sm = spotdetect.SpotDetectionMethods(psf_queue.ctx)

    with Context(psf_queue.ctx.smlm) as lq_ctx:
        q = sm.CreateLocalizationQueue(imgshape, psf_queue, sdcfg, calib, sumframes=sumframes, ctx=lq_ctx)
        numframes = 0
        for fr,img in movie:
            q.PushFrame(img)
            numframes += 1
       
        while q.NumFinishedFrames() < numframes//sumframes:
            time.sleep(0.1)
    
    dt = time.time() - t0
    print(f"Processed {numframes} frames in {dt} seconds. {numframes/dt:.3f} fps")



def create_calib_obj(gain,offset,imgshape,ctx):
    if type(offset)==str:
        print(f'using mean values from {offset} as camera offset')
        offset=read_tiff.get_tiff_mean(offset)
    
    if( type(offset)==np.ndarray):
        gain = np.ones(imgshape)*gain
        calib = GainOffsetImage_Calib(gain, offset, ctx)
    else:
        calib = GainOffset_Calib(gain, offset, ctx) 
    
    return calib


def localize(fn, cfg, output_file=None, progress_cb=None, estimate_sigma=False):
    sigma = cfg['sigma']
    roisize = cfg['roisize']
    threshold = cfg['threshold']
    gain = cfg['gain']
    offset = cfg['offset']
    startframe = cfg['startframe'] if 'startframe' in cfg else 0
    maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
    sumframes = 1
    
    with SMLM() as smlm, Context(smlm) as ctx:
        imgshape = read_tiff.tiff_get_image_size(fn)

        gaussian = Gaussian(ctx)
            
        spotDetectorConfig = spotdetect.SpotDetectorConfig(np.mean(sigma), roisize, threshold)

        if estimate_sigma:
            psf = gaussian.CreatePSF_XYIBgSigmaXY(roisize, sigma, True)
        else:
            psf = gaussian.CreatePSF_XYIBg(roisize, sigma, True)

        queue = PSF_Queue(psf, batchSize=1024)
        
        calib = create_calib_obj(gain,offset,imgshape,ctx)

        process_movie(imgshape, spotDetectorConfig, calib, queue, 
                           read_tiff.tiff_read_file(fn, startframe, maxframes, progress_cb), sumframes)

        queue.WaitUntilDone()
        
        if progress_cb is not None:
            if not progress_cb(None,None): return None,None
        
        r = queue.GetResults()
        r.SortByID() # sort by frame numbers
        
        print(f"Filtering {len(r.estim)} spots...")
        minX = 2.1
        minY = 2.1
        r.FilterXY(minX,minY,roisize-minX-1, roisize-minY-1)
        r.Filter(np.where(r.iterations<50)[0])
        
        nframes = np.max(r.ids)+1 if len(r.ids)>0 else 1
        print(f"Num spots: {len(r.estim)}. {len(r.estim) / nframes} spots/frame")
        
        if output_file is not None:
            r.SaveHDF5(output_file, imgshape)
            
        
        return r,imgshape
    
    

if __name__ == "__main__":
    fn = '/data/hd/sim1_200mw_7_MMStack_Pos0_merge.ome.tif'

    sigma = 1.83
    roisize = 9
    threshold = 3
    gain = 2.2
    offset = read_tiff.get_tiff_mean('/data/GainImages/1bg_1_MMStack_Pos0.ome.tif')
    maxframes = -1
    
    cfg={
        'sigma':sigma,
        'roisize':roisize,
        'threshold':threshold,
        'gain':gain,
        'offset':offset
    }
    
    locs_file = os.path.splitext(fn)[0]+".hdf5"
    results, imgshape = localize(fn, cfg, output_file=locs_file)

#    from utils.link_locs import estimate_on_time
 #   estimate_on_time(locs_file)
    
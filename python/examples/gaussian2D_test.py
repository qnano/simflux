# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:16:22 2018

@author: jcnossen1
"""
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smlmlib.gaussian import Gaussian
from smlmlib.base import SMLM
from smlmlib.spotdetect import SpotDetectorConfig
from smlmlib.context import Context
import smlmlib.util as su
import math
from scipy.stats import poisson
import time
import os
import utils.localizations as loc
import tqdm

import smlmlib.spotdetect as spotdetect
from smlmlib.calib import GainOffset_Calib
from smlmlib.calib import GainOffsetImage_Calib
from smlmlib.context import Context
from smlmlib.psf_queue import PSF_Queue

debugMode=True

def process_movie_slow(imgshape, sdcfg, calib, psf_queue:PSF_Queue, movie):
    t0 = time.time()
    
    sm = spotdetect.SpotDetectionMethods(psf_queue.ctx)

    numframes = 0
    for fr,img in movie:
        rois,cornerYX,scores=sm.ProcessFrame(img, sdcfg, 1000, calib=calib)
        psf_queue.Schedule(rois,roipos=cornerYX,ids=np.ones(len(rois))*fr)
        
        numframes += 1

    dt = time.time() - t0
    print(f"Processed {numframes} frames in {dt} seconds. {numframes/dt:.3f} fps")



def process_movie(imgshape, sdcfg, calib, psf_queue:PSF_Queue, movie):
    t0 = time.time()
    
    sm = spotdetect.SpotDetectionMethods(psf_queue.ctx)

    with Context(psf_queue.ctx.smlm) as lq_ctx:
        q = sm.CreateLocalizationQueue(imgshape, psf_queue, sdcfg, calib, sumframes=1, ctx=lq_ctx)
        numframes = 0
        for fr,img in movie:
            q.PushFrame(img)
            numframes += 1

        while q.NumFinishedFrames() < numframes:
            time.sleep(0.1)
    
    dt = time.time() - t0
    print(f"Processed {numframes} frames in {dt} seconds. {numframes/dt:.3f} fps")


def localize(mov, sigma, roisize, minIntensity, ctx, calib, fn):
    imgshape = mov[0].shape

    gaussian = Gaussian(ctx)
    spotDetectorConfig = spotdetect.SpotDetectorConfig(sigma, roisize, minIntensity)
    
    psf = gaussian.CreatePSF_XYIBg(roisize, sigma, True)
    queue = PSF_Queue(psf, batchSize=1024)
    
    fn(imgshape, spotDetectorConfig, calib, queue, enumerate(mov))

    queue.WaitUntilDone()

    r = queue.GetResults()
    
    nframes =  np.max(r.ids)+1 if len(r.ids)>0 else 1
    print(f"Num spots: {len(r.estim)}. {len(r.estim) / nframes} spots/frame")

    cfg = { 'sigma':psfSigma,
       'roisize':roisize,
       'maxSpotsPerFrame':2000,
       'detectionMinIntensity':minIntensity}
    return loc.from_psf_queue_results(r, cfg, [0,0,imgshape[1],imgshape[0]], '')[0]
        
        
def generate_movie(gaussian, emitterList, numframes=100, imgsize=512, intensity=500, bg=2, sigma=1.5):
    frames = np.zeros((numframes, imgsize, imgsize), dtype=np.uint16)
    emitters = np.array([[e[0], e[1], sigma, sigma, intensity] for e in emitterList])

    for f in range(numframes):
        frame = bg * np.ones((imgsize, imgsize), dtype=np.float32)
        frames[f] = gaussian.Draw(frame, emitters)

    return frames

   
psfSigma = 2
roisize = 10
w = 100
numframes = 2
minIntensity = 4
gain=0.5
offset=0

x_pos = np.linspace(20, w-20, 3)+.5
y_pos = np.linspace(20, w-20, 3)+.5
print(f"# spots: {len(x_pos)*len(y_pos)}")
emitter_x, emitter_y = np.meshgrid(x_pos, y_pos)
emitters = np.vstack((emitter_x.flatten(), emitter_y.flatten())).T

emitters[:,[0,1]] += np.random.uniform(-2,2,size=(len(emitters),2))

def localize_old(mov,ctx):
    cfg = { 'sigma':psfSigma,
       'roisize':roisize,
       'maxSpotsPerFrame':2000,
       'detectionMinIntensity':minIntensity,
       'detectionMaxIntensity':1e6,
       'offset':offset,
       'gain':gain
       }

    r = loc.LocResultList()
    r.process_images(cfg, iter(mov), len(mov), ctx)
    return r

with SMLM(debugMode=debugMode) as smlm, Context(smlm) as ctx:
    gaussian = Gaussian(ctx)

    #r.save_picasso_hdf5('test.hdf5')
    mov = generate_movie(gaussian, emitters, numframes=numframes, imgsize=w, sigma=psfSigma,intensity=500,bg=10)
    smp = np.random.poisson(mov)

    calib = GainOffset_Calib(gain, offset, ctx) 

    r1 = localize_old(smp, ctx)
    r2 = localize(smp, psfSigma, roisize, minIntensity, ctx, calib, process_movie_slow)
    r3 = localize(smp, psfSigma, roisize, minIntensity, ctx, calib, process_movie)
    
    dc = loc.DataColumns
    
    for r in [r1,r2,r3]:
        print(f"Result count: {len(r.data)}")
        
        sel = np.argsort(r.data[:,dc.X])        
        print(r.data[sel][:,[dc.FRAME, dc.X, dc.Y, dc.I, dc.BG]])
        
        
        
        
        
 




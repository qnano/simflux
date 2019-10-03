# -*- coding: utf-8 -*-

"""
Using a PSF_Queue and PSF_Image_Queue_2D to run the whole PALM pipeline
"""
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from smlmlib.gaussian import Gaussian
from smlmlib.base import SMLM
import smlmlib.spotdetect as spotdetect
import math
import time
import smlmlib.util as smlmutil

from smlmlib.context import Context

from smlmlib.psf_image_queue import PSF_Image_Queue_2D
from smlmlib.psf_queue import PSF_Queue


def generate_storm_movie(gaussian, emitterList, numframes=100, imgsize=512, intensity=500, bg=2, sigma=1.5, p_on=0.1):
    frames = np.zeros((numframes, imgsize, imgsize), dtype=np.uint16)
    emitters = np.array([[e[0], e[1], sigma, sigma, intensity] for e in emitterList])

    on_counts = np.zeros(numframes, dtype=np.int32)

    for f in range(numframes):
        frame = bg * np.ones((imgsize, imgsize), dtype=np.float32)
        frame_emitters = emitters * 1
        on = np.random.binomial(1, p_on, len(emitters))
        frame_emitters[:, 4] *= on

        frame = gaussian.Draw(frame, frame_emitters)
        frames[f] = frame
        on_counts[f] = np.sum(on)

    return frames, on_counts


def process_movie(img_queue: PSF_Image_Queue_2D, movie):
    t0 = time.time()
    
    psf_queue = img_queue.psf_queue

    img_queue.Start()
    
    img_queue.SetVerbose(False)

    repeats=50
    for r in range(repeats):
        for f in range(len(mov)):
            img_queue.PushFrame(mov[f])
        
    while not img_queue.IsIdle() or not psf_queue.IsIdle():
        if img_queue.IsIdle():
            psf_queue.Flush()
            
#        print( img_queue.GetStreamState())
 
        if True:           
            print("Images in process queue: {0}. Images in copy queue: {1}. PSF Queue: {2} spots. Results: {3} spots".format(
                    img_queue.GetProcessQueueLength(),
                    img_queue.GetCopyQueueLength(),
                    psf_queue.GetQueueLength(),
                    psf_queue.GetResultCount()))
        time.sleep(1)

    dt = time.time() - t0
    numframes = len(mov)*repeats
    print(f"Processed {numframes} frames in {dt} seconds. {numframes/dt:.3f} fps")

    return psf_queue.GetResults().estim

psfSigma = 1.8
roisize = 10
w = 256
N = 2000
numframes = 100
R = np.random.normal(0, 0.2, size=N) + w * 0.3
angle = np.random.uniform(0, 2 * math.pi, N)
emitters = np.vstack((R * np.cos(angle) + w / 2, R * np.sin(angle) + w / 2)).T

with SMLM(debugMode=False) as smlm:
    with Context(smlm) as ctx:
        gaussian = Gaussian(ctx)
        np.random.seed(0)
        mov, on_counts = generate_storm_movie(gaussian, emitters, numframes, imgsize=w, sigma=psfSigma, p_on=20 / N)
        mov = np.random.poisson(mov)
        
        plt.imshow(mov[0])
    
        spotDetectorConfig = spotdetect.SpotDetectorConfig(psfSigma, roisize, minIntensity=2)
        
        psf = gaussian.CreatePSF_XYIBg(roisize, psfSigma, True)
        queue = PSF_Queue(psf, batchSize=256)
        img_queue = PSF_Image_Queue_2D(queue, mov[0].shape, spotDetectorConfig, cudaStreamCount=3)
        estim = process_movie(img_queue,mov)

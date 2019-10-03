# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import time

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian

from smlmlib.psf import PSF
from smlmlib.psf_queue import PSF_Queue

from smlmlib.calib import sCMOS_Calib


# Calibration coefficients
s0_x = 1.68114296
gamma_x = -1.47879879
d_x = 2.72178031e01
A_x = 2.23656865e-04

s0_y = 1.40361319e00
gamma_y = 3.22323250e01
d_y = 2.12416436e01
A_y = 1.00000000e-05

calib = gaussian.Gauss3D_Calibration([s0_x, gamma_x, d_x, A_x], [s0_y, gamma_y, d_y, A_y])


def test_queue_output(ctx: Context, psf:PSF, theta):
    numspots = 5
    theta_ = np.repeat(theta,numspots,axis=0)
    smp = np.random.poisson(psf.ExpectedValue(theta_))
    estim1 = psf.ComputeMLE(smp)[0]
    
    q = PSF_Queue(psf, batchSize=4, numStreams=3)
    q.Schedule(smp, ids=np.arange(numspots))
    q.WaitUntilDone()
    results = q.GetResults()
    results.SortByID()
    print(estim1)
    print(results.estim)
    print(results.ids)
    
    assert( np.sum( np.abs(results.estim-estim1) ) < 0.01)
    
    
def test_psf_speed(ctx: Context, smp_psf:PSF, est_psf:PSF, theta, batchSize=1024*4,repeats=100):
    img = smp_psf.ExpectedValue(theta)
    smp = np.random.poisson(img)
    plt.figure()
    plt.imshow(smp[0])

    queue = PSF_Queue(est_psf, batchSize=batchSize, numStreams=4)
    n = 10000
    repd = np.ascontiguousarray(np.repeat(smp,n,axis=0),dtype=np.float32)

    t0 = time.time()
    total = 0
    for i in range(repeats):
        queue.Schedule(repd)
        results = queue.GetResults()
        total += n
        
    queue.Flush()
    while not queue.IsIdle():
        time.sleep(0.05)

    results = queue.GetResults()
#    print(results.CRLB())
    t1 = time.time()
    
    queue.Destroy()
            
    print(f"Finished. Processed {total} in {t1-t0:.2f} s. {total/(t1-t0):.1f} spots/s")


with SMLM(debugMode=True) as smlm:

    
    with Context(smlm) as ctx:
    
        sigma=1.5
        w = 512
        roisize=7
        theta=[[roisize//2, roisize//2, 1000, 5]]
        g_api = gaussian.Gaussian(ctx)
        psf = g_api.CreatePSF_XYIBg(roisize, sigma, True)
        scmos = sCMOS_Calib(ctx, np.zeros((w,w)), np.ones((w,w)), np.ones((w,w))*5)
        psf_sc = g_api.CreatePSF_XYIBg(roisize, sigma, True, scmos)
                
    #    test_queue_output(ctx, psf, theta)
        print('2D Gaussian fit:')
        test_psf_speed(ctx,psf,psf,theta)

        """        
        print('2D Gaussian fit + GLRT:')
        psf_glrt = PSF.GLRT_PSF_Create(psf, ctx)
        test_psf_speed(ctx,psf_glrt,psf_glrt,theta)
    
        theta=[[roisize//2, roisize//2, 1000, 3]]
        print('2D Gaussian fit + sCMOS:')
        test_psf_speed(ctx,psf_sc,psf_sc,theta)
    
        print('2D Gaussian fit + sCMOS + GLRT:')
        psf_sc_glrt = PSF.GLRT_PSF_Create(psf_sc, ctx)
        test_psf_speed(ctx,psf_sc_glrt,psf_sc_glrt,theta)
    
        print('Phasor:')
        phasor_est= PSF.PhasorEstimator(roisize, ctx)
        test_psf_speed(ctx,psf,phasor_est,theta,batchSize=1024*10, repeats=1000)
    
        print('COM:')
        com_est = PSF.CenterOfMassEstimator(roisize, ctx)
        test_psf_speed(ctx,psf,com_est,theta,batchSize=1024*10, repeats=1000)
        """
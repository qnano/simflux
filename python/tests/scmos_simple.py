# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import smlmlib.util as su
from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian
from smlmlib.calib import sCMOS_Calib
import matplotlib.pyplot as plt
    
roisize=8
imgshape = [roisize*2,roisize*2]
 
offset = np.zeros(imgshape)
gain = np.ones(imgshape)
variance = np.ones(imgshape)*1

with SMLM(debugMode=True) as smlm:
    with Context(smlm) as ctx:
        g = gaussian.Gaussian(ctx)
    
        sigma=2.5
        theta=[[roisize/2, roisize/2, 10000, 2]]
        roipos = [1,1] # Bad pixel should now show up upper right corner

        scmos_calib = sCMOS_Calib(ctx, offset, gain, variance)
        psf_sc = g.CreatePSF_XYIBg(roisize, sigma, True, scmos_calib)

        ev_sc = psf_sc.ExpectedValue(theta,roipos=[roipos])
        
        crlb_sc=psf_sc.CRLB(theta)
        
        smp=np.random.poisson(ev_sc)
        estim,diag,traces=psf_sc.ComputeMLE(smp,roipos=[roipos])
        print(estim)
        
        plt.plot(traces[0][:,2])
        print(traces[0][:,2])
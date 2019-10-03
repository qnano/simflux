
import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import time

import smlmlib.util as su
from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian

from smlmlib.calib import sCMOS_Calib

from smlmlib.simflux import SIMFLUX
from smlmlib.psf import PSF
import tqdm


def plot_traces(traces, theta, axis=0):
    plt.figure()
    for k in range(len(traces)):
        plt.plot(np.arange(len(traces[k])),traces[k][:,axis])
        plt.plot([len(traces[k])],[theta[k,axis]],'o')

def estimate_precision(psf:PSF, psf_mle:PSF, thetas, photons):
    prec = np.zeros((len(photons),thetas.shape[1]))
    bias = np.zeros((len(photons),thetas.shape[1]))
    crlb = np.zeros((len(photons),thetas.shape[1]))
    Iidx = psf.FindThetaIndex('I')
    
    print(f"I index: {Iidx}")
    for i in tqdm.trange(len(photons)):
        thetas_ = thetas*1
        thetas_[:, Iidx] = photons[i]
        roipos = np.random.randint(0,20,size=(len(thetas_), psf.indexdims))
        smp = psf.GenerateSample(thetas_,roipos=roipos)
        estim,diag,traces = psf_mle.ComputeMLE(smp,roipos=roipos)
        
        if i == len(photons)-1 and len(traces[0])>1:
            plot_traces(traces[:20], thetas_[:20], axis=2)
            
        crlb_ = psf.CRLB(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)

#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb


with SMLM(debugMode=False) as smlm, Context(smlm) as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.8
    roisize=9
    numspots = 10000
    theta=[[roisize/2, roisize/2, 1000, 15]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-2,-1,size=(numspots,2))
    
    useCuda=True
    
    g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
    phasor = PSF.PhasorEstimator(roisize, ctx)
    com = PSF.CenterOfMassEstimator(roisize, ctx)
    
    photons = np.logspace(2, 4, 20)

    data = [
        (g_psf, "2D Gaussian + COM", estimate_precision(g_psf, com, theta, photons)),
        (g_psf, "2D Gaussian + Phasor", estimate_precision(g_psf, phasor, theta, photons)),
        (g_psf, "2D Gaussian + MLE", estimate_precision(g_psf, g_psf, theta, photons)),
        ]

    axes=['x']
    axes_unit=['nm', 'nm','photons','photons/pixel']
    axes_scale=[100, 100, 1, 1]
    for i,ax in enumerate(axes):
        plt.figure()
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.FindThetaIndex(ax)
            plt.plot(photons,axes_scale[i]*prec[:,ai],label=f'Precision {name}')
            plt.plot(photons,axes_scale[i]*crlb[:,ai],'--', label=f'CRLB {name}')

        plt.title(f'{ax} axis - roisize {roisize}x{roisize}')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Signal intensity [photons]')
        plt.ylabel(f"{ax} [{axes_unit[i]}]")
        plt.grid(True)
        plt.legend()

        plt.figure()
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.FindThetaIndex(ax)
            plt.plot(photons,bias[:,ai],label=f'Bias {name}')

        plt.title(f'{ax} axis - roisize {roisize}x{roisize}')
        plt.grid(True)
        plt.xscale("log")
        plt.legend()
        plt.show()




# -*- coding: utf-8 -*-

# Generate a random noise / gain / variance map
# Simulate spots
# Localize with and without sCMOS correction


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

from smlmlib.psf import PSF
from smlmlib.simflux import SIMFLUX
import tqdm

import scipy.io as sio

roisize=16
imgshape = [roisize*2,roisize*2]

calib_path = '../../../SMLM/data-jochem/'
gain = sio.loadmat(calib_path + 'gainmap_10ms.mat')['gainmap']
showoffset = sio.loadmat(calib_path + 'OffsetMap_10ms_IJ.mat')['offsetmap']
variance = sio.loadmat(calib_path + 'VarianceMap_10ms_IJ.mat')['variancemap']
variance /= gain**2

gain = np.ones(variance.shape)
offset = np.zeros(variance.shape)



pitch = 221/65
k = 2*np.pi/pitch

mod = np.array([
           [0, k,   0.95, 0, 1/6],
           [k, 0, 0.95, 0, 1/6],
           [0, k,   0.95, 2*np.pi/3, 1/6],
           [k, 0, 0.95, 2*np.pi/3, 1/6],
           [0, k,   0.95, 4*np.pi/3, 1/6],
           [k, 0, 0.95, 4*np.pi/3, 1/6]
          ])

def plot_traces(psf, theta, smp):   
    estim,diag,traces = psf.ComputeMLE(smp)

    plt.figure()
    axis = 2
    for i,tr in enumerate(traces):
        p = plt.plot(tr[:, axis])
        col = p[0].get_color()
        plt.plot(len(tr), [theta[i, axis]], "o", color=col)
        
def log_hist(data):
    plt.hist(data)
#    plt.hist(data, bins=np.logspace(np.log10(0.1),np.log10(1.0), 50))
    plt.yscale('log')
    plt.show()

def estimate_precision(psf_ev:PSF, psf_sc:PSF, psf_est:PSF, thetas, roipos, photons, mlefn=None):
    prec = np.zeros((len(photons),thetas.shape[1]))
    bias = np.zeros((len(photons),thetas.shape[1]))
    crlb = np.zeros((len(photons),thetas.shape[1]))
    Iidx = psf_ev.FindThetaIndex('I')
    Bidx = psf_ev.FindThetaIndex('bg')
    
    #print(f"I index: {Iidx}")
    for i in tqdm.trange(len(photons)):
        thetas_ = thetas*1
        thetas_[:, Iidx] = photons[i]
        ev = psf_ev.ExpectedValue(thetas_,roipos=roipos)

        empty = thetas*1
        empty[:, Iidx] = 0
        empty[:, Bidx] = 0
        readnoise = psf_sc.ExpectedValue(empty, roipos=roipos)
        
        smp = np.random.poisson(ev) + np.random.normal(0,np.sqrt(readnoise)) 
        estim,diag,traces = psf_est.ComputeMLE(smp,initial=thetas,roipos=roipos)
            
        crlb_ = psf_est.CRLB(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)
    
#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb
    
with SMLM(debugMode=False) as smlm, Context(smlm) as ctx:
    g = gaussian.Gaussian(ctx)

    sigma = 2
    numspots = 1000
    theta=[[roisize/2, roisize/2, 200, 8]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-1/2,1/2,size=(numspots,2))
    sf_roipos=np.random.randint(0,512-roisize,size=(numspots,3))
    sf_roipos[:,0]=0
    roipos=sf_roipos[:,[1,2]]

    scmos_calib = sCMOS_Calib(ctx, offset, gain, variance)
    
    useCuda=True
            
    g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
    g_psf_sc = g.CreatePSF_XYIBg(roisize, sigma, useCuda, scmos_calib)

    sf = SIMFLUX(ctx)
    sf_psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, len(mod),True,False)
    sf_psf_sc = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, len(mod),True,False,scmos_calib)
    
#    plot_traces(sf_psf, theta, g_psf_sc.GenerateSample(theta[:20]))
    plot_traces(sf_psf_sc, theta, sf_psf_sc.GenerateSample(theta[:20]))


    def draw_plots(data):
        axes=['x','y', 'I', 'bg']
        axes_unit=['nm', 'nm','photons','photons/pixel']
        axes_scale=[100, 100, 1, 1]
        for i,ax in enumerate(axes):
            plt.figure()
            for psf,name,(prec,bias,crlb) in data:
                ai = psf.FindThetaIndex(ax)
                plt.plot(photons,axes_scale[i]*prec[:,ai],label=f'Precision {name}')
                plt.plot(photons,axes_scale[i]*crlb[:,ai],'--', label=f'CRLB {name}')

            plt.title(f'{ax} axis')
            plt.xscale("log")
            plt.xlabel('Signal intensity [photons]')
            plt.ylabel(f"{ax} [{axes_unit[i]}]")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()

            plt.figure()
            for psf,name,(prec,bias,crlb) in data:
                ai = psf.FindThetaIndex(ax)
                plt.plot(photons, axes_scale[i]* bias[:,ai],label=f'Bias {name}')

            plt.title(f'{ax} axis')
            plt.grid(True)
            plt.ylabel(f"{ax} [{axes_unit[i]}]")
            plt.xlabel('Signal intensity [photons]')
            plt.xscale("log")
            plt.legend()
            plt.show()



    photons = np.logspace(2, 4, 20)

    if False:
        g_data = [
            (g_psf, "2D Gaussian", estimate_precision(g_psf, g_psf, g_psf, theta, roipos, photons)),
            (g_psf, "2D Gaussian (sCMOS)", estimate_precision(g_psf, g_psf_sc, g_psf_sc, theta, roipos, photons)),
            (g_psf, "2D Gaussian (Regular on sCMOS sample)", estimate_precision(g_psf, g_psf_sc, g_psf, theta, roipos, photons)),
            (sf_psf, "2D Gaussian SIMFLUX", estimate_precision(sf_psf, sf_psf, sf_psf, theta, sf_roipos, photons)),
            ]
        draw_plots(g_data)

    if True:
        sf_data = [
            (sf_psf, "SIMFLUX (Poisson-only MLE)", estimate_precision(sf_psf, sf_psf, sf_psf, theta, sf_roipos, photons)),
            (sf_psf, "SIMFLUX (MLE with sCMOS correction)", estimate_precision(sf_psf, sf_psf_sc, sf_psf_sc, theta, sf_roipos, photons)),
            (sf_psf, "SIMFLUX (Poisson-MLE on sCMOS sample)", estimate_precision(sf_psf, sf_psf_sc, sf_psf, theta, sf_roipos, photons)),
            ]
        
        draw_plots(sf_data)
        

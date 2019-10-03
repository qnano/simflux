 # -*- coding: utf-8 -*-

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

def estimate_precision(psf:PSF, psf_mle:PSF, thetas, photons, mlefn=None):
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
        if mlefn is not None:
            estim = mlefn(smp, roipos)
        else:
            estim,diag,traces = psf_mle.ComputeMLE(smp,roipos=roipos)
            
        crlb_ = psf.CRLB(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)

#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb

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
    
    
calib = gaussian.Gauss3D_Calibration.from_file('../../data/simulated_as_gaussian.npy')

with SMLM(debugMode=False) as smlm:
    with Context(smlm) as ctx:
        g = gaussian.Gaussian(ctx)
    
        sigma=2
        roisize=16
        numspots = 10000
        theta=[[roisize/2, roisize/2, 10009, 5]]
        theta=np.repeat(theta,numspots,0)
        theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
        
        useCuda=True
        
    #    with g.CreatePSF_XYZIBg(roisize, calib, True) as psf:
        g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
        g_s_psf = g.CreatePSF_XYIBgSigma(roisize, sigma+1, useCuda)
        g_sxy_psf = g.CreatePSF_XYIBgSigmaXY(roisize, [sigma+1, sigma+1], useCuda)
        g_z_psf = g.CreatePSF_XYZIBg(roisize, calib, useCuda)
        
        calib_size = (roisize+20, roisize+20)
        offset = np.zeros(calib_size)
        gain = np.ones(calib_size)
        var = np.ones(calib_size)*1
        scmos_calib = sCMOS_Calib(ctx, offset, gain, var)
        g_psf_noisy = g.CreatePSF_XYIBg(roisize, sigma, True, scmos=scmos_calib)
        
        #com_psf = COM(ctx).CreatePSF(roisize, g_psf)

        theta_z=np.zeros((numspots,5)) # x,y,z,I,bg
        theta_z[:,[0,1]]=theta[:,[0,1]]
        theta_z[:,4] = 2

        theta_sig=np.zeros((numspots,6))
        theta_sig[:,0:4]=theta
        theta_sig[:,[4,5]]=sigma
        theta_sig[:,[4,5]] += np.random.uniform(-0.5,0.5,size=(numspots,2))
                
        s = SIMFLUX(ctx)
        sf_psf = s.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, len(mod), True)

        def sf_mle(smp, roipos):
            mod_=np.repeat([mod],len(smp),axis=0)
            roiposXY = roipos[:,[2,1]]
            startPatterns = roipos[:,0]
            return s.SIMFLUX_ASW_ComputeMLE(smp, mod_, s.Params(roisize, len(mod), sigma), 
                                            startPatterns, roiposXY=roiposXY, cuda=useCuda)[0]
        
        photons = np.logspace(2, 4, 50)

        if True:
            data = [
                (g_psf, "2D Gaussian with readnoise", estimate_precision(g_psf_noisy,g_psf_noisy, theta, photons)),
                (g_psf, "2D Gaussian", estimate_precision(g_psf, g_psf, theta, photons)),
                (g_psf, "Non-readnoise fit on readnoise sample", estimate_precision(g_psf_noisy, g_psf, theta, photons)),
       #         (sf_psf, "SIMFLUX", estimate_precision(sf_psf, theta, photons)),
                #(g_z_psf, 'Z-Fitted astig. 2D Gauss',estimate_precision(g_z_psf, theta_z, photons)),
                #(g_s_psf, '2D Gauss Sigma', estimate_precision(g_s_psf, theta_sig[:,:5], photons)),
                #(g_sxy_psf, '2D Gauss Sigma XY', estimate_precision(g_sxy_psf, theta_sig, photons)),
                ]
   #             addplot(ax,g_s_psf,theta_sig[:,:5],photons, '2D Gaussian (sigma) MLE')
    #            addplot(ax,g_sxy_psf,theta_sig,photons, '2D Gaussian (sigma xy) MLE')
  #              addplot(ax,sf_psf,theta,photons, 'SIMFLUX MLE', mlefn=sf_mle)

            axes=['x','I']
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
                plt.yscale("log")
                plt.xlabel('Signal intensity [photons]')
                plt.ylabel(f"{ax} [{axes_unit[i]}]")
                plt.grid(True)
                plt.legend()

                plt.figure()
                for psf,name,(prec,bias,crlb) in data:
                    ai = psf.FindThetaIndex(ax)
                    plt.plot(photons,bias[:,ai],label=f'Bias {name}')

                plt.title(f'{ax} axis')
                plt.grid(True)
                plt.xscale("log")
                plt.legend()
                plt.show()

        if False:
            for ax in [0,2]:
                plt.figure()
                theta_z = np.zeros((numspots,5))
                theta_z[:,0:4] = theta
                addplot(ax,g_z_psf,theta_z,photons, '2D Gaussian (xyz) MLE')

                axname = g_z_psf.ThetaFormat().split(',')[ax]
                plt.title(f'Axis {axname}')
                plt.xscale("log")
                plt.yscale("log")
                plt.grid(True)
                plt.legend()
                plt.show()
        
    

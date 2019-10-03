
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


def phasor_localize(roi):
    fx = np.sum(np.sum(roi,0)*np.exp(-2j*np.pi*np.arange(roi.shape[1])/roi.shape[1]))
    fy = np.sum(np.sum(roi,1)*np.exp(-2j*np.pi*np.arange(roi.shape[0])/roi.shape[0]))
            
    #Get the size of the matrix
    WindowPixelSize = roi.shape[1]
    #Calculate the angle of the X-phasor from the first Fourier coefficient in X
    angX = np.angle(fx)
    if angX>0: angX=angX-2*np.pi
    #Normalize the angle by 2pi and the amount of pixels of the ROI
    PositionX = np.abs(angX)/(2*np.pi/WindowPixelSize)
    #Calculate the angle of the Y-phasor from the first Fourier coefficient in Y
    angY = np.angle(fy)
    #Correct the angle
    if angY>0: angY=angY-2*np.pi
    #Normalize the angle by 2pi and the amount of pixels of the ROI
    PositionY = np.abs(angY)/(2*np.pi/WindowPixelSize)
    #Calculate the magnitude of the X and Y phasors by taking the absolute
#    value of the first Fourier coefficient in X and Y
    MagnitudeX = np.abs(fx)
    MagnitudeY = np.abs(fy)
    
    return PositionX,PositionY

with SMLM(debugMode=False) as smlm, Context(smlm) as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.6
    roisize=7

    psf = g.CreatePSF_XYIBg(roisize, sigma, False)
    theta = [4,4,10000,10]

    plt.figure()
    img = psf.GenerateSample([theta])
    plt.figure()
    plt.set_cmap('inferno')
    plt.imshow(img[0])
    
    phasor_localize(img[0])
    
    com = PSF.CenterOfMassEstimator(roisize,ctx)
    phasor = PSF.PhasorEstimator(roisize,ctx)
    
    com_estim = com.ComputeMLE(img)[0]
    print(f"COM: {com_estim}")
    
    phasor_estim = phasor.ComputeMLE(img)[0]
    print(f"Phasor: {phasor_estim}")
    
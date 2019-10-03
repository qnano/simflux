import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian
from smlmlib.util import imshow_hstack
from smlmlib.psf import PSF

def CheckDeriv(psf:PSF, theta):
    nderiv,ev=psf.NumDeriv(theta,eps=1e-6)
    deriv,ev=psf.Derivatives(theta)

    maxerr = np.max( np.abs(deriv-nderiv), (-1,-2) )
    print(f"PSF {psf.ThetaFormat()}, max {np.max(deriv)}, min: {np.min(deriv)}: Deriv-NumDeriv: {maxerr}")

    plt.figure()
    imshow_hstack(deriv[0] - nderiv[0])

with SMLM(debugMode=True) as smlm:
    
    with Context(smlm) as ctx:
        g = gaussian.Gaussian(ctx)
    
        for cuda in [False]:
            print(f"CUDA = {cuda}")
            sigma=2
            roisize=12

            psf = g.CreatePSF_XYIBg(roisize, sigma, cuda)
            
            theta = [[4, 4, 1000, 3]]
            img = psf.ExpectedValue(theta)
            plt.figure()
            plt.set_cmap('inferno')
    
            smp = np.random.poisson(img)
            plt.imshow(smp[0])
            
            psf_sigma = g.CreatePSF_XYIBgSigma(roisize, sigma, cuda)
                        
            theta_s = [[4,4,1000,3,sigma]]
            img2 = psf_sigma.ExpectedValue(theta_s)

            CheckDeriv(psf, theta)
#            CheckDeriv(psf_sigma)
            
            
            print(f"PSF Sigma crlb: {psf_sigma.CRLB(theta_s)}")

            theta = psf_sigma.ComputeMLE(smp)[0]
            print(theta)
            
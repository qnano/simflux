import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian
from smlmlib.util import imshow_hstack


with SMLM(debugMode=True) as smlm, Context(smlm) as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.5
    roisize=9

    psf = g.CreatePSF_XYIBg(roisize, sigma, False)
    theta = [roisize/2,roisize/2,1000,1]

    plt.figure()
    deriv_err = psf.NumDeriv([theta])[0]-psf.Derivatives([theta])[0]

    img = psf.ExpectedValue([theta])
    plt.figure()
    plt.set_cmap('inferno')
    plt.imshow(img[0])

    smp = np.random.poisson(img)
    estim, _, trace = psf.ComputeMLE(smp)
    print(estim)
    
    plt.figure()
    plt.bar(np.arange(9),np.sum(smp[0],0))
    theta=estim[0]
    x=np.linspace(0,9)
    plt.plot(x, theta[2] * np.exp( -((x-theta[0])/(np.sqrt(2)*sigma)) ** 2) / (sigma*np.sqrt(2*np.pi)) + theta[3] * roisize,
             'k',linewidth=6)
#            plt.plot(np.arange(9),np.sum(img[0],0),'r',linewidth=5)            
    crlb = psf.CRLB(estim)
    print(f"2D Gaussian CRLB: {crlb}")
    
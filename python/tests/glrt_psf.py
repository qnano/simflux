# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian
from smlmlib.util import imshow_hstack
from smlmlib.psf import PSF

from scipy.stats import norm

with SMLM(debugMode=True) as smlm, Context(smlm) as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.5
    roisize=9

    psf = g.CreatePSF_XYIBg(roisize, sigma, True)
    psf = PSF.GLRT_PSF_Create(psf, ctx, None)

    theta = [roisize/2,roisize/2,50,5]

    img = psf.ExpectedValue([theta])

    smp = np.random.poisson(img)
    plt.figure()
    plt.set_cmap('inferno')
    plt.imshow(smp[0])

    estim, diag, trace = psf.ComputeMLE(smp)

    ll_on = diag[:,0]
    ll_off = diag[:,1]
    
    #float Tg = 2.0f*(h1_ll - h0_ll);	// Test statistic (T_g)
	#float pfa = 2.0f * cdf(-sqrt(Tg)); // False positive probability (P_fa)

    Tg = 2*(ll_on-ll_off)
    pfa = 2 * norm.cdf(-np.sqrt(Tg))
    
    print(f"p(false positive): {pfa}")
    
    print(estim)
    print(diag)
    
    
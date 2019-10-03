# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import time

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian
from smlmlib.util import imshow_hstack
from smlmlib.psf_queue import PSF_Queue
from smlmlib.simflux import SIMFLUX
from smlmlib.gaussian import Gaussian
from smlmlib.psf import PSF

mod = np.array([
           [0, 1.8,   0.95, 0, 1/6],
           [1.9, 0, 0.95, 0, 1/6],
           [0, 1.8,   0.95, 2*np.pi/3, 1/6],
           [1.9, 0, 0.95, 2*np.pi/3, 1/6],
           [0, 1.8,   0.95, 4*np.pi/3, 1/6],
           [1.9, 0, 0.95, 4*np.pi/3, 1/6]
          ])

with SMLM(debugMode=False) as smlm:
    with Context(smlm) as ctx:
        sf = SIMFLUX(ctx)
        g = Gaussian(ctx)
    
        sigma=1.5
        roisize=10
        
#        psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, 6, simfluxEstim=False)
#        psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, 6, simfluxEstim=True)
        psf = g.CreatePSF_XYIBg(roisize, sigma, True)
        psfcr = PSF.CopyROI_Create(psf,ctx)

        n = 20
        repeats = 10
        theta=np.repeat([[roisize//2, roisize//2, 1000, 3]],n,axis=0)
        np.random.seed(0)
        roipos = np.random.randint(0,len(mod),size=(n,psf.indexdims))
        
        img = psf.ExpectedValue(theta,roipos=roipos)
        smp = np.random.poisson(img)

        def runqueue():
            queue = PSF_Queue(psfcr, batchSize=4, numStreams=4)
            t0 = time.time()
            total = 0
            for i in range(repeats):
                ids = i*n+np.arange(n)
                smp[:,0,0] = ids
                queue.Schedule(smp,roipos=roipos,ids=ids)
                total += n
            queue.WaitUntilDone()
            t1 = time.time()
            print(f"Finished. Processed {total} in {t1-t0} s. {total/(t1-t0):.1f} spots/s")
            r = queue.GetResults()
            
            r.SortByID(isUnique=True)
            r.rois = r.diagnostics.reshape((len(r.ids), *psf.samplesize))
            return r

        def compute():
            estim,_,_ = psf.ComputeMLE(np.repeat(smp,repeats,0), roipos=np.repeat(roipos,repeats,0))
            return estim

#        ac = compute()
#        bc = compute()
#        assert(np.all(np.equal(ac, bc)))

        a = runqueue()
        b = runqueue()
 
        if True:
            for k in range(1):
                a_ids = a.rois[:,0,0]
                b_ids = b.rois[:,0,0]
                chk = a_ids != b_ids
                nz = np.nonzero(chk)[0]
            assert(np.all(np.equal(a.estim, b.estim)))
    
            # -*- coding: utf-8 -*-


import sys
sys.path.append("simflux")

import simflux.process_tiff as process_tiff
import numpy as np
import urllib.request

import os

pixelsize = 65 # nm/pixel

mod = np.zeros((6,5))

config = { 
   'roisize':9,
   'maxSpotsPerFrame':2000,
   'detectionMaxIntensity':20000,
   'startframe': 0,
   'detectionMinIntensity': 6,
   'sigma':1.663,
   'offset':0,
   'gain':1
}

sigma_sim = 1.663

# Simple worm-like chain simulated model
path = 'object_wlc_15.tif' 

if not os.path.exists(path):
    print('Beginning file download with urllib2...')
    
    url = 'http://homepage.tudelft.nl/f04a3/object_wlc_15.tif'
    urllib.request.urlretrieve(url, path)

    

spotfilter = ('moderror', 0.012)
pattern_frames = [[0,2,4],[1,3,5]] 

process_tiff.process(path, config, 
    pattern_frames,
    pixelsize=pixelsize,
    num_phase_bins=10,
    estimate_angle_bins=1,
    plot_ffts=True, 
    show_plots =False,
    draw_silm_compare=True,
    drift_correct=None,
    debugMode=False,
    fix_depths=None,
    fix_phase_shifts=None,
    freq_minmax=[1.7,1.9],
    spotfilter=spotfilter)


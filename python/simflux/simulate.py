# Simulate SILM measurements and save them to a TIFF file
import sys

sys.path.append("..")

import numpy as np
from smlmlib.base import SMLM
from smlmlib.gaussian import Gaussian
import matplotlib.pyplot as plt
import math
import tifffile
import os
from util.blinking_spots import spots as blinking_spots
import util.excitation_model as exc_model
import tqdm
import silm_process_tiff
import pickle

sigma = 1.8304

def generate_tiff(path, emitters, width, numframes, intensityMin, intensityMax, allmod, bg, psf):
    os.makedirs(os.path.split(path)[0], exist_ok=True)

    spots = np.random.uniform([0,0,intensityMin],[0,0,intensityMax],size=(len(emitters),3))
    spots[:,[0,1]] += emitters

    blinking = blinking_spots(spots, numframes,avg_on_time=80,on_fraction=0.02)

    roisize = psf.roisize
    
    with tifffile.TiffWriter(path) as tif:
        
        for f in tqdm.trange(numframes):
            mod = allmod[f%len(allmod)]
            ev = np.ones((width,width),dtype=np.float32) * bg
                        
            spots = blinking[f]*1
            q = exc_model.compute(spots[:,0],spots[:,1],mod)
            spots[:,2] *= q
            
            theta = np.zeros((len(spots),4))
            theta[:,[0,1,2]] = spots
            roipos = theta[:,[0,1]].astype(int) - roisize//2
            theta[:,[0,1]] -= roipos
            
            spotev = psf.ExpectedValue(theta)
            for i,pos in enumerate(roipos):
                ev[pos[0]:pos[0]+roisize,pos[1]:pos[1]+roisize] += spotev[i]
            
#            ev = loc.draw_spots(ev, spots[:,0], spots[:,1], spots[:,2], sigma, sigma, 1, smlm)
            
            if f == 0:
                plt.imshow(ev)
                
            smp = np.random.poisson(ev)
            tif.save(np.ascontiguousarray(smp, dtype=np.uint16))

    return spots, blinking

    
def generate(path, density, width, numframes, allmod, smlm: SMLM):    
    
    x_pos = np.linspace(50, width-50, 60)+.5
    y_pos = np.linspace(50, width-50, 60)+.5
    print(f"# spots: {len(x_pos)*len(y_pos)}")
    emitter_x, emitter_y = np.meshgrid(x_pos, y_pos)
    grid_emitters = np.vstack((emitter_x.flatten(), emitter_y.flatten())).T
    grid_emitters += np.random.uniform([-3,-3],[3,3],size=grid_emitters.shape)

    N=100
    R = np.random.normal(0, 0.2, size=N) + width * 0.1
    angle = np.linspace(0, 2 * math.pi, N)
    circle_emitters = np.vstack((R * np.cos(angle) + width / 2, R * np.sin(angle) + width / 2)).T

    emitters = np.vstack((grid_emitters,circle_emitters))

    roisize=12

    plt.figure()    
    plt.scatter(emitters[:,0],emitters[:,1],marker='.')
#    plt.draw()
    plt.show()
    
    bg=10

    with Gaussian(smlm).CreatePSF_XYIBg(roisize, sigma, cuda=True) as psf:
        return generate_tiff(path, emitters, width, numframes, 500, 2000,allmod, bg, psf)
    
if __name__ == '__main__':

    path = '../../data/sim-silm/'
    
    true_mod= np.array([
               [0, 1.8,   0.95, 0, 1/6],
               [1.85, 0, 0.95, 0, 1/6],
               [0, 1.8,   0.95, 2*np.pi/3, 1/6],
               [1.85, 0, 0.95, 2*np.pi/3, 1/6],
               [0, 1.8,   0.95, 4*np.pi/3, 1/6],
               [1.85, 0, 0.95, 4*np.pi/3, 1/6]
              ])


    # Generate or Test?
    if False:
            
        with SMLM() as smlm:
            densities = [0.02]#np.linspace(0.005,0.05, 10)
            
            for i in range(len(densities)):
                spots,blinking = generate(path + f'sim{i}.tiff', densities[i], 400, 12000, true_mod, smlm)
            
                d = {'density': densities[i],
                     'sigma': sigma,
                     'mod': true_mod,
                     'spots': spots,
                     'blinking': blinking
                     }
            
                with open(path+f"sim{i}-true.pickle","wb") as f:
                    pickle.dump(d, f)
                    
    else:

        fn = path+f'sim0.tiff'        

        # Configuration for spot detection and 2D Gaussian fitting
        cfg = { 'sigma':sigma,
           'roisize':10,
           'maxSpotsPerFrame':2000,
           'detectionMinIntensity':30,
           'detectionMaxIntensity':2001,
           'offset':0,
           'gain':1
           }
        
        pixelsize = 65 # nm/pixel
        
        # Modulations format: [ kx, ky, depth, phase, power ]
        mod = np.zeros((6,5))        
        
        mod=true_mod
        # Estimate modulation patterns and process the frames in a non-sliding window way 
        # i.e. frame: [0:5], [6:11], [12:17] ...
        spotlist, img, estmod, silm_result, g2d_result = silm_process_tiff.estimate_patterns(fn, cfg, mod, 
                            pixelsize=pixelsize,
                            num_phase_bins=10,
                            estimate_angle_bins=1,
                            plot_ffts=True, 
                            show_plots=False,
                            draw_silm_compare=True,
                            debugMode=False,
                            freq_minmax=[1.75,1.95],
                            silm_min_intensity=30,
                            fix_phase_shifts=None)
            
        true_pitch = 2*np.pi/np.sqrt(true_mod[:,0]**2+true_mod[:,1]**2) * pixelsize
        est_pitch = 2*np.pi/np.sqrt(estmod[:,0]**2+estmod[:,1]**2)  * pixelsize
        for k in range(6):
            print(f"True pitch: {true_pitch[k]:.5f}. Est pitch: {est_pitch[k]:.5f}. Error: {est_pitch[k]-true_pitch[k]} ({(est_pitch[k]-true_pitch[k])/true_pitch[k]*100} %)")
        

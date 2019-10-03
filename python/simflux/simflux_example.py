import sys
sys.path.append("..")
sys.path.append(".")
import simflux.process_tiff as process_tiff
import numpy as np
import tifffile
#import util.spotlist as spotlist

pixelsize = 65 # nm/pixel

# Modulations format: [ kx, ky, depth, phase, power ]
mod = np.zeros((6,5))

def get_tiff_mean(fn):
    mov = tifffile.imread(fn)
    return np.mean(mov,0)


if __name__ == '__main__':

    # Configuration for spot detection and 2D Gaussian fitting
    basecfg = { 
        'sigma':[1.829,1.723],
       'roisize':9,
       'maxSpotsPerFrame':2000,
       'detectionMinIntensity':6,
       'detectionMaxIntensity':20000,
       'startframe': 0,
#       'spotfilter': ('minfilter-firstlast', 30),
       'spotfilter': ('moderror', 0.012),
       'patternFrames': [[0,2,4],[1,3,5]],
       'gain' : 2.2,
       'offset': 100
 #      'maxframes': 10000
    }

    sigma_sim = 1.663
    
    files = [
        # Format is: TIFF Path, detection threshold, number of patterns, gain, offset
#        ('O:/45/4Phase_3/4Phase_3.tif', { 'detectionMinIntensity': 10, 'patternFrames': [[0,2,4,6],[1,3,5,7]], 'sigma':[1.885,1.764], 'spotfilter':('moderror',0.12)} ),
 #       ('/data/sim-silm/object_wlc_15.tif',{'detectionMinIntensity': 6,'sigma':[1.663,1.663],'offset':0,'gain':1 }),
#        ('/data/sim-silm/object_grid_dxdy.tif',{'detectionMinIntensity': 5,'sigma':[sigma_sim,sigma_sim],'offset':0,'gain':1 }),
#        ('/data/sim-silm/object_sfig15_1800_8.tif',{'detectionMinIntensity': 5,'sigma':[sigma_sim,sigma_sim],'offset':0,'gain':1 }),
        ('O:/7-23-2019/Pos5/1_merge.tif', {'startframe': 6000, 'spotfilter': ('moderror', np.sqrt(0.04))}),
        ('/data/sim4_1/sim4_1.tif', {'spotfilter': ('moderror', np.sqrt(0.012))}),
 #       ('O:/PaintTubules_8nt_3-13-2019/1_1/1_1_MMStack_Pos0.ome.tif', {'spotfilter': ('moderror', np.sqrt(0.05)),'fix_depths':0.9})
#    	('/dev/simflux/data/PaintTubules/Pos5/1_merge.tif', { 'detectionMinIntensity': 5,  'startframe': 6000 } ),
#        ('/dev/simflux/data/06082019/object_filamentousWLC_05082019_rho1E3.tif', 5, 6, 1, 0),
 #       ('/dev/simflux/data/06082019/object_filamentousWLC_20192507_rho1E3.tif', 5, 6, 1, 0),
#        ('/dev/simflux/data/06082019/object_filamentousWLC_20192407_rho10E3.tif', 5, 6, 1, 0),
 #       ('/dev/simflux/data/06082019/object_filamentousWLC_20192407_rho20E3.tif', 5, 6, 1, 0),
#		('/dev/simflux/data/7-23-2019/Pos5/1_merge.tif', 5, 6, 2.1, 100),
  #       ('/dev/simflux/data/PaintTubules/1_1/1_1_MMStack_Pos0.ome.tif', 8, 6, 2.1, 100)
      ]
    

    for path, cfg2 in files:  
#        IBg, lr = spotlist.tiff_localize(path, mod, False, {**cfg,'detectionMinIntensity':detectionMinIntensity})
        
       # if offset != 0:
        #    offset = '/data/GainImages/1bg_1_MMStack_Pos0.ome.tif'
        
        cfg_ = { **basecfg, **cfg2 }
        
        pattern_frames = cfg_['patternFrames']
        
        spotfilter = cfg_['spotfilter']
        del cfg_['spotfilter'] # make sure this doesnt invalidate the cache
        
        drift_correct=None
        if 'drift_correct' in cfg_:
            drift_correct=cfg_['drift_correct']
            del cfg_['drift_correct']
            
        fix_depths=None
        if 'fix_depths' in cfg_:
            fix_depths=cfg_['fix_depths']
            del cfg_['fix_depths']
                    
        r = process_tiff.process(path, cfg_, 
                            pattern_frames,
                            pixelsize=pixelsize,
                            num_phase_bins=10,
                            estimate_angle_bins=1,
                            plot_ffts=True, 
                            show_plots =False,
                            draw_silm_compare=True,
                            drift_correct=drift_correct,
                            debugMode=False,
                            fix_depths=fix_depths,
                            fix_phase_shifts=None,
                            freq_minmax=[1.7,1.9],
#                            spotfilter=['minfilter-all', 30])
                            spotfilter=spotfilter
							)
                            #spotfilter=['minfilter-firstlast', 30])
        if r is not None:
            spotlist, image, mod, loc_simflux, loc_sum = r
        
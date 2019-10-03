# -*- coding: utf-8 -*-
import h5py 
import yaml
import numpy as np
import os


def load(fn):
    with h5py.File(fn, 'r') as f:
        locs = f['locs']
       
        estim = np.zeros((len(locs),4))
        estim[:,0] = locs['x']
        estim[:,1] = locs['y']
        estim[:,2] = locs['photons']
        estim[:,3] = locs['bg']
        
        crlb = np.zeros((len(locs),4))
        crlb[:,0] = locs['lpx']
        crlb[:,1] = locs['lpy']
        crlb[:,2] = locs['lI']
        crlb[:,3] = locs['lbg']
        
        info_fn = os.path.splitext(fn)[0] + ".yaml" 
        with open(info_fn, "r") as file:
            if hasattr(yaml, 'unsafe_load'):
                obj = yaml.unsafe_load(file)
            else:
                obj = yaml.load(file)
            imgshape=np.array([obj['Height'],obj['Width']])
        
        return estim,locs['frame'],crlb, imgshape
        
        
def save(fn, xyIBg, crlb, framenum, imgshape, sigmaX, sigmaY, extraColumns=None):   
    print(f"Saving hdf5 to {fn}")
    with h5py.File(fn, 'w') as f:
        dtype = [('frame', '<u4'), 
                 ('x', '<f4'), ('y', '<f4'), 
                 ('photons', '<f4'), 
                 ('sx', '<f4'), ('sy', '<f4'), 
                 ('bg', '<f4'), 
                 ('lpx', '<f4'), ('lpy', '<f4'), 
                 ('lI', '<f4'), ('lbg', '<f4'), 
                 ('ellipticity', '<f4'), 
                 ('net_gradient', '<f4')]
        
        if extraColumns is not None:
            for k in extraColumns.keys():
                dtype.append((k,extraColumns[k].dtype,extraColumns[k].shape[1:]))
        
        locs = f.create_dataset('locs', shape=(len(xyIBg),), dtype=dtype)
        locs['frame'] = framenum
        locs['x'] = xyIBg[:,0]
        locs['y'] = xyIBg[:,1]
        locs['photons'] = xyIBg[:,2]
        locs['bg'] = xyIBg[:,3]
        locs['sx'] = sigmaX
        locs['sy'] = sigmaY
        locs['lpx'] = crlb[:,0]
        locs['lpy'] = crlb[:,1]
        locs['lI'] = crlb[:,2],
        locs['lbg'] = crlb[:,3]
        locs['net_gradient'] = 0
        
        if extraColumns is not None:
            for k in extraColumns.keys():
                locs[k] = extraColumns[k] 
                
        info =  {'Byte Order': '<',
                 'Camera': 'Dont know' ,
                 'Data Type': 'uint16',
                 'File': fn,
                 'Frames': np.max(framenum)+1,
                 'Width': imgshape[1],
                 'Height': imgshape[0]
                 }

        info_fn = os.path.splitext(fn)[0] + ".yaml" 
        with open(info_fn, "w") as file:
            yaml.dump(info, file, default_flow_style=False) 

    return fn
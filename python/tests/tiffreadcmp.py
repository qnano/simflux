# -*- coding: utf-8 -*-

import tifffile
import imageio
import tqdm
import numpy as np

maxframes = 2000

def tiff_read_tifffile(fn, callback):
    with tifffile.TiffFile(fn) as tif:
        p0 = tif.pages[0]
        height = p0.shape[0]
        width = p0.shape[1]
        numframes = len(tif.pages)
        print(f"{fn}: width: {width}, height: {height}")

        if numframes>maxframes: numframes=maxframes
    
        for index in tqdm.trange(numframes):
            callback(index,tif.pages[index].asarray())


def tiff_read_imageio(fn, callback):
    with imageio.get_reader(fn) as reader:

        numframes=reader.get_length()
        
        if numframes>maxframes: numframes=maxframes
        
        for index in tqdm.trange(numframes):
            callback(index, reader.get_next_data())

               
fn = '../../../SMLM/data/sim4_1/sim4_1.tif'

means = []

def store_mean(index,img):
    means.append(np.mean(img))

print('tifffile:')
tiff_read_tifffile(fn, store_mean)
print('imageio:')
tiff_read_imageio(fn, store_mean)


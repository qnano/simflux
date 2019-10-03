# Throw away some photons to simulate lower excitation

import tifffile
import tqdm
import os
import numpy as np
import sys
from scandir import scandir
from multiprocessing import Pool


def split_image(img, psplit, add_bg):
    img_int = img.astype(int)
    img_int[img_int<0] = 0
    return np.random.binomial(img_int, psplit) + np.random.poisson(add_bg*psplit,size=img.shape)

# First nbg frames of input: Estimate median bg, generate sample
# f >= nbg < total-nbg:  
def split_tiff(tiffinput, tiffoutput, psplit, offset=0, gain=1, nbg=100, bgfile= None):
    print (tiffoutput)
    print(bgfile)
    with tifffile.TiffWriter(tiffoutput) as out_tif, tifffile.TiffWriter(bgfile) as bg_tif:
        with tifffile.TiffFile(tiffinput) as in_tif:
            total = len(in_tif.pages)

            buffer = np.zeros((nbg, * in_tif.pages[0].shape), dtype=np.int)
            for f in range(nbg):
                buffer[f] = in_tif.pages[f].asarray()
            bufindex = 0
            bg = (np.median(buffer,0)-offset)/gain
 
            for f in range(total):               
#            for f in tqdm.trange(total):
                if f < nbg:
                    img = buffer[f]
                else:
                    img = in_tif.pages[f].asarray()
                    buffer[bufindex] = img; bufindex+=1
                    if(bufindex==nbg): 
                        bufindex=0
                        bg = ((np.median(buffer,0)-offset)/gain).astype(np.uint16)
                        bg_tif.save(bg)

                img = (img-offset)/gain                
                out_tif.save(np.ascontiguousarray(split_image(img, psplit, bg), dtype=np.uint16))
                sys.stdout.write(f"\rframe {f}/{total} ({f/total*100:.2f}%)")
    print()
    
def _process(args):
    path, outdir, psplit, offset, gain, nbg = args
    print(f"pid={os.getpid()}: {path}")
    filename = os.path.split(path)[1]
    outfile = outdir + filename
    bgfile = outdir + "bg/" + filename 
    os.makedirs(outdir + "bg/",exist_ok=True)
    split_tiff(path,outfile,psplit,offset,gain,nbg,bgfile)
    
            
def split_dir(inputdir, outputdir, psplit, nbg, offset=0, gain=1):
    
    params = []
    
    def cb(fn):
        args=[ fn, outputdir, psplit, offset, gain, nbg]
        #params.append(args)
        _process(args)
        
    scandir(inputdir, "*.tif", cb)
    
#    p = Pool(8)
#    p.map(_process, params)
    

if __name__ == "__main__":
    split_dir('O:/mod/', 'O:/mod-psplit/',  0.5, nbg=1000, offset=100.2,gain=1/0.47)
#    split_tiff('../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-1_0.tif',
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0.tif', 0.5, 100.2, 1/0.47, 300,
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0-bg.tif')
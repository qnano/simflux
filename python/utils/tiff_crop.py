
import tifffile
import os
import numpy as np
import sys
from scandir import scandir
from multiprocessing import Pool


# First nbg frames of input: Estimate median bg, generate sample
# f >= nbg < total-nbg:  
def crop_tiff(tiffinput, tiffoutput, xymin, xymax):
    print (tiffoutput)
    with tifffile.TiffWriter(tiffoutput) as out_tif:
        with tifffile.TiffFile(tiffinput) as in_tif:
            total = len(in_tif.pages)
 
            for f in range(total):               
                img = in_tif.pages[f].asarray()
                img = img[xymin[1]:xymax[1], xymin[0]:xymax[0]]
                out_tif.save(img)
                sys.stdout.write(f"\rframe {f}/{total} ({f/total*100:.2f}%)")
    print()
    
def _process(args):
    path, outdir, xymin, xymax = args
    print(f"pid={os.getpid()}: {path}")
    filename = os.path.split(path)[1]
    outfile = outdir + filename
    crop_tiff(path,outfile,xymin,xymax)
    
            
def crop_dir(inputdir, outputdir, xymin, xymax):
        
    def cb(fn):
        args=[ fn, outputdir, xymin, xymax]
        #params.append(args)
        _process(args)

    os.makedirs(outputdir,exist_ok=True)
        
    scandir(inputdir, "*.tif", cb)
    
#    p = Pool(8)
#    p.map(_process, params)
    

if __name__ == "__main__":
    crop_dir('O:/mod/', 'O:/mod-crop2/',  [50,50],[350,350])

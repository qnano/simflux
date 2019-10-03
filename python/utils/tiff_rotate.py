
import tifffile
import os
import numpy as np
import sys
from scandir import scandir

def transpose_tiff(tiffinput, tiffoutput):
    print (tiffoutput)
    os.makedirs(os.path.split(tiffoutput)[0],exist_ok=True)
    with tifffile.TiffWriter(tiffoutput,bigtiff=True) as out_tif:
        with tifffile.TiffFile(tiffinput) as in_tif:
            total = len(in_tif.pages)
 
            for f in range(total):               
                img = in_tif.pages[f].asarray()
                img = np.transpose(img)
                out_tif.save(img)
                sys.stdout.write(f"\rframe {f+1}/{total} ({f/total*100:.2f}%)")
    print()
    
def _process(args):
    path, outdir, xymin, xymax = args
    print(f"pid={os.getpid()}: {path}")
    filename = os.path.split(path)[1]
    outfile = outdir + filename
    transpose_tiff(path,outfile,xymin,xymax)
    
            
def transpose_dir(inputdir, outputdir, xymin, xymax):
        
    def cb(fn):
        args=[ fn, outputdir, xymin, xymax]
        #params.append(args)
        _process(args)

    os.makedirs(outputdir,exist_ok=True)
        
    scandir(inputdir, "*.tif", cb)
    
#    p = Pool(8)
#    p.map(_process, params)
    

if __name__ == "__main__":
#    transpose_dir('O:/mod/', 'O:/mod-crop2/',  [50,50],[350,350])
#    split_tiff('../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-1_0.tif',
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0.tif', 0.5, 100.2, 1/0.47, 300,
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0-bg.tif')
    
    pass

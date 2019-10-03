
import tifffile
import tqdm
import os
import numpy as np
import sys
import fnmatch

def scandir(path, pat, cb):
    for root, dirs, files in os.walk(path):
        head, tail = os.path.split(root)
        for file in files:
            if fnmatch.fnmatch(file, pat):
                fn = os.path.join(root, file)
                cb(fn)

def sumframes(tiffinput, tiffoutput, numframes):
    print (tiffoutput)
    with tifffile.TiffWriter(tiffoutput) as out_tif:
        with tifffile.TiffFile(tiffinput) as in_tif:
            total = len(in_tif.pages)
            
            framesum = in_tif.pages[0].asarray()*0

            n = 0 
            for f in range(total):               
                framesum += in_tif.pages[f].asarray()
                n += 1
                if (n==numframes):
                    out_tif.save(framesum.astype(dtype=np.uint16))
                    sys.stdout.write(f"\rframe {f}/{total} ({f/total*100:.2f}%)")
                    n=0
                    framesum *= 0
    
    print()

def _process(args):
    path, outdir, numframes = args
    print(f"pid={os.getpid()}: {path}")
    filename = os.path.split(path)[1]
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir + filename
    sumframes(path,outfile,numframes)
    

def sumframes_dir(inputdir, outputdir, numframes):
    params = []
    
    def cb(fn):
        args=[ fn, outputdir, numframes]
        #params.append(args)
        _process(args)
        
    scandir(inputdir, "*.tif", cb)
    
#    p = Pool(8)
#    p.map(_process, params)
    

if __name__ == "__main__":
    sumframes_dir('O:/mod/', 'O:/mod-sumframes/',  6)
#    split_tiff('../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-1_0.tif',
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0.tif', 0.5, 100.2, 1/0.47, 300,
#              '../../../SMLM/data/gattaquant 80nm thirdshift/80nm-3rdshift-psplit-1_0-bg.tif')
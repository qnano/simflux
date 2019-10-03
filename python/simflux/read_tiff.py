import numpy as np
import os
import tqdm
import tifffile


def get_tiff_mean(fn):
    mov = tifffile.imread(fn)
    return np.mean(mov,0)


def tiff_get_filenames(fn):
    ext = os.path.splitext(fn)
       
    if fn.find('.ome.tif') >= 0:
        fmt =  "_{0}.ome.tif"
        basefn = fn.replace('.ome.tif','')
    else:
        fmt = "_X{0}.tif"
        basefn = fn.replace('.tif' ,'')
        
    
    files=[fn]
    i = 1
    while True:
        file = basefn+fmt.format(i)
        if os.path.exists(file):
            files.append(file)
            i+=1
        else:
            break
    return files


def tiff_read_file(fn, startframe, maxframes=-1, update_cb=None):
    if update_cb is not None:
        update_cb("Enumerating tif files",0)
    numframes = 0
    filenames = tiff_get_filenames(fn)
    framecounts = []
    
    if maxframes>=0:
        maxframes += startframe
        
    for name in filenames:
        with tifffile.TiffFile(name) as tif:
            framecount = len(tif.pages)
            if maxframes>=0 and framecount + numframes > maxframes:
                framecounts.append(maxframes-numframes)
                numframes = maxframes
                break
            else:
                framecounts.append(framecount)
                numframes += framecount
            
    numframes -= startframe

    print(f'reading tiff file: {maxframes}/{numframes}')

    if maxframes>=0 and maxframes<numframes: 
        numframes=maxframes
        
    with tqdm.tqdm(total=numframes) as pbar:
        index = 0
        for t,name in enumerate(filenames):
            if startframe>=framecounts[t]:
                startframe-=framecounts[t]
                continue
            with tifffile.TiffFile(name) as tif:
                fn_ = name.replace(os.path.dirname(fn)+"/",'')

                for i in np.arange(startframe,framecounts[t]):
                    pbar.update(1)
                    if index % 20 == 0 and update_cb is not None:
                        if not update_cb(f"Reading {fn_} - frame {index}/{numframes}", index/numframes):
                            print(f"Aborting reading tiff file..")
                            return
                        
                    yield index, tif.pages[i].asarray()
                    index += 1
                startframe=0


def tiff_get_image_size(fn):
    with tifffile.TiffFile(fn) as tif:
        shape = tif.pages[0].asarray().shape
        return shape

def tiff_get_movie_size(fn):
    names = tiff_get_filenames(fn)
    numframes = 0
    for name in names:
        with tifffile.TiffFile(name) as tif:
            numframes += len(tif.pages)
            shape = tif.pages[0].asarray().shape
    return shape, numframes
    
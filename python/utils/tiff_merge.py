# -*- coding: utf-8 -*-
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

def merge_tiff(inputs, tiffoutput):
    with tifffile.TiffWriter(tiffoutput,bigtiff=True) as out_tif:
        for tiffinput in inputs:
            with tifffile.TiffFile(tiffinput) as in_tif:
                total = len(in_tif.pages)

                for f in range(total):               
                    img = in_tif.pages[f].asarray()
                    out_tif.save(img)
                    sys.stdout.write(f"\rframe {f}/{total} ({f/total*100:.2f}%)")

if __name__ == "__main__":
 #   merge_tiff(['../../data/80nm/80nm-3rdshift-2_0.tif', '../../data/80nm/80nm-3rdshift-2_1.tif'  ], 
               #'../../data/80nm/80nm-3rdshift-2.tif')
#    merge_tiff(['O:/mod/ld-80nm-2_0.tif', 'O:/mod/ld-80nm-2_1.tif'], 'O:/mod/ld-80nm-2.tif')
#    merge_tiff(['O:/mod/ld-80nm-1_0.tif', 'O:/mod/ld-80nm-1_1.tif'], 'O:/mod/ld-80nm-1.tif')
#    merge_tiff(['O:/mod/80nm-thirdshift-2_0.tif', 'O:/mod/80nm-thirdshift-2_1.tif'], 
 #              'O:/mod/80nm-thirdshift-2.tif')

    """
        merge_tiff(['O:/sim1_1/sim1_1_MMStack_Pos0.ome.tif',
                    'O:/sim1_1/sim1_1_MMStack_Pos0_1.ome.tif',
                    'O:/sim1_1/sim1_1_MMStack_Pos0_2.ome.tif',
                    'O:/sim1_1/sim1_1_MMStack_Pos0_3.ome.tif'],
                    'O:/sim1_1/sim1_1.tif')
    """

    merge_tiff(['O:/70FPS Munich Data/3Phase_3/3Phase_3_MMStack_Pos0.ome.tif', 
                'O:/70FPS Munich Data/3Phase_3/3Phase_3_MMStack_Pos0_1.ome.tif',
                'O:/70FPS Munich Data/3Phase_3/3Phase_3_MMStack_Pos0_2.ome.tif',
                'O:/70FPS Munich Data/3Phase_3/3Phase_3_MMStack_Pos0_3.ome.tif'],
        'O:/70FPS Munich Data/3Phase_3/3Phase_3_merged.tif')

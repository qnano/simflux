import numpy as np
import tifffile


def load_movie(filename, num_frames=None, skip=0):
    with tifffile.TiffFile(filename, movie=True) as tiff:
        images = tiff.asarray()
        if num_frames is not None:
            images = images[skip : skip + num_frames, :, :]
        return np.squeeze(images)

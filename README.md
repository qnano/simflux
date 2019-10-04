# SIMFLUX Readme

SIMFLUX is implemented using C++/CUDA 10.1 and Python, running on Windows 10. A version with the binary already build can be found [here](http://homepage.tudelft.nl/f04a3/simflux-publication-release.zip)

## Installing python

If the binary is already there, you only need to install python and some python packages. It is recommended to use anaconda 3 or [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Make sure to get python >= 3.6 on 64 bit:
https://repo.anaconda.com/archive/Anaconda3-2019.07-Windows-x86_64.exe

```bash
conda create -n simflux_env python
conda activate simflux_env
conda install tqdm matplotlib scikit-image pyyaml scikit-image scikit-learn h5py
conda install -c conda-forge tifffile
```

## Building C++ / CUDA library

(The order is important here)
- Install Visual studio 2017 or 2019 Community Edition: https://visualstudio.microsoft.com/downloads/
- Open visual studio. Menu "Tools / Get Tools and Features"
- Install the Visual studio 2015 build toolset. We're using the Visual Studio 2015 build tools because we ran into cuda compiler issues at 2017.

- Install CUDA Toolkit 10.1 https://developer.nvidia.com/cuda-downloads
If you install this before visual studio 2017 and the 2015 toolset, you need to reinstall it to make sure the CUDA BuildCustomizations are installed.

- Download the external libraries below and extract in the root. There should be a folder "external" in the repository root.
https://drive.google.com/open?id=1E7f7xqNLwqtj4_-2bjJ56T6kbzsdCpDQ

- Open the SMLM.sln solution file. Don't upgrade the projects.


## Testing

python/simflux_example.py can be used as a basic test and example of how to run the code. It will automatically download some sample data of around 200MB.

```
cd python
python simflux_example.py
```

The results will be in a folder named "results/object_wlc_15", with various plots and localization files.

- simflux.hdf5 - SIMFLUX localizations
- g2d.hdf5 - Conventional SMLM localizations

Linked localizations (Localizations in subsequent frames merged into one more accurate spot, for FRC calculation):
- simflux-linked.hdf5 - SIMFLUX localizations
- g2d-linked.hdf5 - Conventional SMLM localizations

These HDF5 files are compatible with the [Picasso render application](https://github.com/jungmannlab/picasso) for viewing:
![](http://homepage.tudelft.nl/f04a3/simflux.png)


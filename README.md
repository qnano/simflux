# SIMFLUX Readme

SIMFLUX is implemented using C++/CUDA 10.1 and Python, running on Windows 10. A version with the binary already build can be found here:
TODO: insert link

## Installing python

If the binary is already there, you only need to install python and some python packages. It is recommended to use anaconda 3
Make sure to get python >= 3.6 on 64 bit:
https://repo.anaconda.com/archive/Anaconda3-2019.07-Windows-x86_64.exe

conda create -n simflux_env anaconda
conda activate simflux_env
conda install tqdm matplotlib scikit-image pyyaml
pip install tifffile


## Building C++ / CUDA library

Install CUDA Toolkit 10.1 or higher:
https://developer.nvidia.com/cuda-downloads.

(The order is important here)
- Install Visual studio 2017 Community Edition: https://visualstudio.microsoft.com/downloads/
- Open visual studio. Menu "Tools / Get Tools and Features"
- Install the Visual studio 2015 build toolset

- Install CUDA Toolkit 10.1 https://developer.nvidia.com/cuda-downloads
If you install this before visual studio 2017 and the 2015 toolset, you need to reinstall it to make sure the CUDA BuildCustomizations are installed.

- Download the external libraries below and extract in the root. There should be a folder "external" in the repository root.
https://drive.google.com/open?id=1E7f7xqNLwqtj4_-2bjJ56T6kbzsdCpDQ


## Testing

python/simflux_example.py can be used as a basic test and example of how to run the code. It will automatically download some sample data of around 200MB.

Each contains:
simflux.hdf5 - SIMFLUX localizations
g2d.hdf5 - Conventional SMLM localizations





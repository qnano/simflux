README
----------------------------------------------------------------------------
This readme shows how to process a TIFF file using SIMFLUX

Step 1. Download and install Anaconda for Windows 64 bit
https://www.anaconda.com/distribution/
https://repo.anaconda.com/archive/Anaconda3-2018.12-Windows-x86_64.exe

Step 3. Install CUDA Toolkit 9.2 or higher:
https://developer.nvidia.com/cuda-downloads.

Step 2.
Create a virtual environment for simflux:
You should have an "Anaconda Prompt" somewhere in the windows apps now.
Open this anaconda prompt and run the following to create an anaconda environment named simflux:
conda create -n simflux anaconda
conda activate simflux

Step 4.
Using the open anaconda prompt, install python and anaconda packages:
conda install -c anaconda cudatoolkit 
pip install scipy matplotlib numpy tifffile tqdm

Step 5. 
Using the anaconda prompt, go to the SIMFLUX directory where simflux.py is located.

Step 6. 
Run SIMFLUX. Passing no arguments will show the help:
python simflux.py

The example dataset can be processed with the following: 
python simflux.py data/80nm-15fps-example.tif --sigma 1.83 --gain 2.2 --offset 100 --threshold 20

Step 7.
The results will be stored in {input directory}/results/{tif name}/*.
The HDF5 files contain localizations that can be opened in Picasso Render (https://github.com/jungmannlab/picasso)

Each result directory contains:
simflux.hdf5 - SIMFLUX localizations
g2d.hdf5 - Conventional SMLM localizations



# thermo

This repository is a collection of structure and data processing functions geared towards calculating thermal properties from molecular dynamics simulations. The code is primarily designed to interface with [GPUMD](https://github.com/brucefan1983/GPUMD) with some supporting [LAMMPS](https://lammps.sandia.gov/) code. 

As of now, the development of the code reflects my needs as a researcher and the interface with **GPUMD** is far from complete. The code is also subject to change drastically with **GPUMD** as we determine permanant **GPUMD** features/formats. The [releases](https://github.com/AlexGabourie/thermo/releases) page identifies which **GPUMD** release corresponds to each **thermo** release.

# Documentation:

https://thermomd.readthedocs.io/en/latest/

# Installation:

The current version of *thermo* is built on Python 2 with plans to update to Python 3 by the end of 2019.

To install the most recent version, use the following command:

pip install git+https://github.com/AlexGabourie/thermo.git

This will install all of the dependencies and the thermo package in the virtual environment.

## Installation issues:
- If pyfftw fails to install, run the following line to install it:

sudo apt-get install libfftw3-dev libfftw3-doc

- If you get an error like: 'Failed building wheel for subprocess32,' run the following:

sudo apt-get install python-dev

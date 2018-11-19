import numpy as np
import os

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

#########################################
# Data-loading Related
#########################################

def load_kappa_output(directory=''):
    """
    Loads data from kappa.out GPUMD output file which contains HNEMD kappa

    out keys:\n
    - kx_in
    - kx_out
    - ky_in
    - ky_out
    - kz

    Args:
        directory (str):
            Directory to load 'kappa.out' file from (dir. of simulation)

    Returns:
        out (dict):
            A dictionary with keys corresponding to the columns in 'kappa.out'
    """

    if directory=='':
        kappa_path = os.path.join(os.getcwd(),'kappa.out')
    else:
        kappa_path = os.path.join(directory,'kappa.out')

    with open(kappa_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    out['kx_in'] = np.zeros(len(lines))
    out['kx_out'] = np.zeros(len(lines))
    out['ky_in'] = np.zeros(len(lines))
    out['ky_out'] = np.zeros(len(lines))
    out['kz'] = np.zeros(len(lines))

    for i, line in enumerate(lines):
        nums = line.split()
        out['kx_in'][i] = float(nums[0])
        out['kx_out'][i] = float(nums[1])
        out['ky_in'][i] = float(nums[2])
        out['ky_out'][i] = float(nums[3])
        out['kz'][i] = float(nums[4])

    return out

def load_hac_output(directory=''):
    """
    Loads data from hac.out GPUMD output file which contains the
    heat-current autocorrelation and running thermal conductivity values

    Created for GPUMD-v1.9

    hacf - (ev^3/amu)
    k - (W/m/K)
    t - (ps)

    out keys:\n
    - hacf_xi
    - hacf_xo
    - hacf_x: ave. of i/o components
    - hacf_yi
    - hacf_yo
    - hacf_y: ave of i/o components
    - hacf_z
    - k_xi
    - k_xo
    - k_x: ave of i/o components
    - k_yi
    - k_yo
    - k_y: ave of i/o components
    - k_z
    - k_i: ave of x/y components
    - k_o: ave of x/y components
    - k: ave of all in-plane components
    - t: correlation time

    Args:
        directory (str):
            Directory to load 'hac.out' file from (dir. of simulation)

    Returns:
        out (dict):
            A dictionary with keys corresponding to the columns in 'hac.out'
            with some additional keys for aggregated values (see description)
    """

    if directory=='':
        hac_path = os.path.join(os.getcwd(),'hac.out')
    else:
        hac_path = os.path.join(directory,'hac.out')

    with open(hac_path, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        t = np.zeros((N,1))
        x_ac_i = np.zeros((N,1)) # autocorrelation IN, X
        x_ac_o = np.zeros((N,1)) # autocorrelation OUT, X

        y_ac_i = np.zeros((N,1)) # autocorrelation IN, Y
        y_ac_o = np.zeros((N,1)) # autocorrelation OUT, Y

        z_ac = np.zeros((N,1)) # autocorrelation Z

        kx_i = np.zeros((N,1)) # kappa IN, X
        kx_o = np.zeros((N,1)) # kappa OUT, X

        ky_i = np.zeros((N,1)) # kappa IN, Y
        ky_o = np.zeros((N,1)) # kappa OUT, Y

        kz = np.zeros((N,1)) # kappa, Z

        for i, line in enumerate(lines):
            vals = line.split()
            t[i] = vals[0]
            x_ac_i[i] = vals[1]
            x_ac_o[i] = vals[2]
            y_ac_i[i] = vals[3]
            y_ac_o[i] = vals[4]
            z_ac[i] = vals[5]
            kx_i[i] = vals[6]
            kx_o[i] = vals[7]
            ky_i[i] = vals[8]
            ky_o[i] = vals[9]
            kz[i] = vals[10]

    out = dict()
    # x-direction heat flux autocorrelation function
    out['hacf_xi'] = x_ac_i
    out['hacf_xo'] = x_ac_o
    out['hacf_x'] = x_ac_i + x_ac_o

    # y-direction heat flux autocorrelation function
    out['hacf_yi'] = y_ac_i
    out['hacf_yo'] = y_ac_o
    out['hacf_y'] = y_ac_i + y_ac_o

    # z-direction heat flux autocorrelation function
    out['hacf_z'] = z_ac

    # x-direction thermal conductivity
    out['k_xi'] = kx_i
    out['k_xo'] = kx_o
    out['k_x'] = kx_i + kx_o

    # y-direction thermal conductivity
    out['k_yi'] = ky_i
    out['k_yo'] = ky_o
    out['k_y'] = ky_i + ky_o

    # z-direction thermal conductivity
    out['k_z'] = kz

    # Combined thermal conductivities (isotropic)
    out['k_i'] = (kx_i + ky_i)/2.
    out['k_o'] = (kx_o + ky_o)/2.
    out['k'] = (out['k_x'] + out['k_y'])/2.

    out['t'] = t

    return out

import numpy as np
import os
from scipy.integrate import trapz
from .data import load_shc

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def running_ave(kappa, time):
    """
    Gets running average. Reads and returns the structure input file from GPUMD.

    Args:
        kappa (ndarray): Raw thermal conductivity
        time (ndarray): Time vector that kappa was sampled at

    Returns:
        ndarray: Running average of kappa input
    """

    out = np.zeros(kappa.shape[0])
    for i, t in enumerate(time):
        out[i] = (1./t*trapz(kappa[:i], time[:i]))
    return out

def hnemd_spectral_decomp(dt, Nc, Fmax, Fe, T, A, Nc_conv=None,
                          shc=None, directory=''):
    """
    Computes the spectral decomposition from HNEMD between two groups
    of atoms.

    Args:
        dt (float):
            Sample period (in fs) of SHC method

        Nc (int):
            Maximum number of correlation steps

        Fmax (float):
            Maximum frequency (THz) to compute spectral decomposition to

        Fe (float):
            HNEMD force in (1/A)

        T (float):
            HNEMD run temperature (in K)

        A (float):
            Area (nm^2) that heat flows over

        Nc_conv (int):
            Number of correlations steps to use for calculation

        shc (dict):
            Dictionary from load_shc if already created

        directory (str):
            Directory to load 'shc.out' file from (dir. of simulation)

    Returns:
        dict: Dictionary with the spectral decomposition

    """
    if shc==None:
        shc = load_shc(Nc, directory)

    dt_in_ps = dt/1000. # ps
    nu = np.arange(0.01, Fmax+0.01, 0.01)

    if not Nc_conv == None:
        Nc = Nc_conv

    ki = shc['shc_in']
    ko = shc['shc_out']

    hann = (np.cos(np.pi*np.arange(0,Nc_conv)/Nc_conv)+1)*0.5
    ki = (ki[0:Nc]*np.array([1] + [2]*(Nc-1)).reshape(1,-1))*hann
    ko = (ko[0:Nc]*np.array([1] + [2]*(Nc-1)).reshape(1,-1))*hann

    qi = np.zeros((nu.shape[0], 1))
    qo = np.zeros((nu.shape[0], 1))

    for i, n in enumerate(nu):
        qi[i] = 2*dt_in_ps*np.sum(ki*np.cos(2*np.pi*n*np.arange(0,Nc_conv)*dt_in_ps))
        qo[i] = 2*dt_in_ps*np.sum(ko*np.cos(2*np.pi*n*np.arange(0,Nc_conv)*dt_in_ps))

    convert = 16.0217662
    ki = convert*qi/A/T/Fe
    ko = convert*qo/A/T/Fe

    out = dict()
    out['ki'] = ki
    out['ko'] = ko
    out['nu'] = nu
    return out

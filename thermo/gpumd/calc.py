import numpy as np
import os
from scipy.integrate import cumtrapz
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
    return cumtrapz(kappa, time, initial=0)/time

def hnemd_spectral_decomp(dt, Nc, Fmax, Fe, T, V, df=None, Nc_conv=None,
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

        V (float):
            Volume (A^3) that heat flows over
            
        df (float):
            Spacing of frequencies in output (THz)

        Nc_conv (int):
            Number of correlations steps to use for calculation

        shc (dict):
            Dictionary from load_shc if already created

        directory (str):
            Directory to load 'shc.out' file from (dir. of simulation)

    Returns:
        dict: Dictionary with the spectral decomposition

    """
    if (not type(Nc) == int):
        raise ValueError('Nc must be an int.')
    
    if (not Nc_conv is None):
        if (not type(Nc_conv) == int):
            raise ValueError('Nc_conv must be an int.')
        if (Nc_conv > Nc):
            raise ValueError('Nc_conv must not be greater than Nc.')
        Nc = Nc_conv
    
    if shc==None:
        shc = load_shc(Nc, directory)
        
    if 1000/dt < Fmax:
        raise ValueError('Sampling frequency must be > 2X Fmax.')
        
    if df is None:
        df = 0.01

    dt_in_ps = dt/1000. # ps
    nu = np.arange(0, Fmax+df, df)

    ki = shc['shc_in']
    ko = shc['shc_out']

    hann = (np.cos(np.pi*np.arange(0,Nc)/Nc)+1)*0.5
    ki = (ki[0:Nc]*np.array([1] + [2]*(Nc-1)).reshape(1,-1))*hann
    ko = (ko[0:Nc]*np.array([1] + [2]*(Nc-1)).reshape(1,-1))*hann

    qi = np.zeros((nu.shape[0], 1))
    qo = np.zeros((nu.shape[0], 1))

    for i, n in enumerate(nu):
        qi[i] = 2*dt_in_ps*np.sum(ki*np.cos(2*np.pi*n*np.arange(0,Nc)*dt_in_ps))
        qo[i] = 2*dt_in_ps*np.sum(ko*np.cos(2*np.pi*n*np.arange(0,Nc)*dt_in_ps))

    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    ki = convert*qi/V/T/Fe
    ko = convert*qo/V/T/Fe

    out = dict()
    out['ki'] = ki.squeeze()
    out['ko'] = ko.squeeze()
    out['nu'] = nu.squeeze()
    return out

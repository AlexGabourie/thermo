import numpy as np
import os
from scipy.integrate import trapz

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def running_ave(kappa, time):
    """
    Gets running average

    Reads and returns the structure input file from GPUMD.

    Parameters
    ----------
    arg1 : kappa
        raw thermal conductivity

    arg2 : time
        time vector that kappa was sampled at

    Returns
    -------
    out
        running average of kappa input
    """

    out = np.zeros(kappa.shape[0])
    for i, t in enumerate(time):
        out[i] = (1./t*trapz(kappa[:i], time[:i]))
    return out

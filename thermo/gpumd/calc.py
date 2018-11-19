import numpy as np
import os
from scipy.integrate import trapz

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def running_ave(kappa, time):
    """
    Gets running average

    Reads and returns the structure input file from GPUMD.

    Args:
        kappa (ndarray):
            Raw thermal conductivity

        time (ndarray):
            Time vector that kappa was sampled at

    Returns:
        out (ndarray):
            Running average of kappa input
    """

    out = np.zeros(kappa.shape[0])
    for i, t in enumerate(time):
        out[i] = (1./t*trapz(kappa[:i], time[:i]))
    return out

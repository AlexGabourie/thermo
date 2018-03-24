import pyfftw
import multiprocessing
import numpy as np

def autocorr(f, max_lag):
    '''Computes a fast autocorrelation function and returns up to max_lag'''
    N = len(f)
    d = N - np.arange(N)
    # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
    f = np.lib.pad(f, (0,N), 'constant', constant_values=(0,0))
    fvi = np.zeros(2*N, dtype=type(f[0]))
    fwd = pyfftw.FFTW(f, fvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd()
    inv_arg = fvi*np.conjugate(fvi)
    acf = np.zeros_like(inv_arg)
    rev = pyfftw.FFTW(inv_arg, acf, direction='FFTW_BACKWARD', 
                      flags=('FFTW_ESTIMATE', ), threads=multiprocessing.cpu_count())
    rev()
    acf = acf[:N]/d
    return np.real(acf[:max_lag+1])
    
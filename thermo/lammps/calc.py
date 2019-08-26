import pyfftw
import multiprocessing
import numpy as np
import traceback
import os
import sys
from scipy import integrate
from math import floor
from .data import extract_dt
import scipy.io as sio

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def autocorr(f, max_lag):
    '''
    Computes a fast autocorrelation function and returns up to max_lag

    Args:
        f (ndarray):
            Vector for autocorrelation

        max_lag (float):
            Lag at which to calculate up to

    Returns:
        out (ndarray):
            Autocorrelation vector

    '''
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

def __metal_to_SI( vol, T ):
    '''
    Converts LAMMPS metal units to SI units for thermal conductivity calculations.

    Args:
        vol (float):
            Volume in angstroms^3

        T (float):
            Temperature in K

    Returns:
        out (float):
            Converted value
    '''
    kb = 1.38064852e-23 #m^3*kg/(s^2*K)
    vol = vol/(1.0e10)**3 #to m^3
    #eV^2*ns/(ps^2*angstrom^4) to J^2/(s*m^4)
    to_SI = (1.602e-19)**2.*1.0e12*(1.0e10)**4.0*1000.
    return vol*to_SI/(kb*T**2)

def get_heat_flux(**kwargs):
    '''
    Gets the heat flux from a LAMMPS EMD simulation. Creates a compressed .mat
    file if only in text form. Loads .mat form if exists.

    out keys:\n


    Args:
        **kwargs (dict):\n
        - directory (str):
            This is the directory in which the simulation results are located. If
            not provided, the current directory is used.
        - heatflux_file (str):
            Filename of heatflux output. If not provided 'heat_out.heatflux' is used
        - mat_file (str):
            MATLAB file to load, if exists. If not provided, 'heat_flux.mat' will
            be used. Also used as filename for saved MATLAB file.

    Returns:
        out (dict):\n
        - Jx (list)
        - Jy (list)
        - Jz (list)
        - rate (float)

    '''
    original_dir = os.getcwd()

    try:
        # Get arguments
        directory = kwargs['directory'] if 'directory' in kwargs.keys() else './'
        heatflux_file = kwargs['heatflux_file'] if 'heatflux_file' in \
            kwargs.keys() else os.path.join(directory, 'heat_out.heatflux')
        mat_file = kwargs['mat_file'] if 'mat_file' in kwargs.keys() \
            else os.path.join(directory, 'heat_flux.mat')

        # Check that directory exists
        if not os.path.isdir(directory):
            raise IOError('The path: {} is not a directory.'.format(directory))

        # Go to directory and see if imported .mat file already exists
        os.chdir(directory)
        if os.path.isfile(mat_file) and mat_file.endswith('.mat'):
            return sio.loadmat(mat_file)

        # Continue with the import since .mat file
        if not os.path.isfile(heatflux_file):
            raise IOError('The file: \'{}{}\' is not found.'.format(directory,heatflux_file))

        # Read the file
        with open(heatflux_file, 'r') as hf_file:
            lines = hf_file.readlines()[2:]

        # Get timestep
        rate = int(lines[0].split()[0])

        # read all data
        jx = list()
        jy = list()
        jz = list()
        for line in lines:
            vals = line.split()
            jx.append(float(vals[1]))
            jy.append(float(vals[2]))
            jz.append(float(vals[3]))

        output = {'Jx':jx, 'Jy':jy, 'Jz':jz, 'rate':rate}
        sio.savemat(mat_file, output)
        os.chdir(original_dir)
        return output

    except:
        os.chdir(original_dir)
        print(sys.exc_info()[0])

def get_GKTC(**kwargs):
    '''
    Gets the thermal conductivity vs. time profile using the Green-Kubo formalism.
    thermal conductivity vector and time vector.
    Assumptions with no info given by user:
    dt = 1 fs, vol = 1, T=300, rate=dt, tau=tot_time

    Keyword Arguments: \n
    - directory (string):
        This is the directory in which the simulation results are located.
        If not provided, the current directory is used.

    - T (float):
        This is the temperature at which the equlibrium simulation was run at.
        If not provided, T=300 is used. Units are in [K]

    - vol (float):
        This is the volume of the simulation system.
        If not provided, vol=1 is used. Units are [angstroms^3]

    - log (string):
        This is the path of the log file. This is only used if the *dt* keyword is not provided
        as it tries to extract the timestep from the logs

    - dt (float):
        This is the timestep of the green-kubo part of the simulation.
        If not provided, dt=1 fs is used. units are in [ps]

    - rate (int):
        This is the rate at which the heat flux is sampled. This is in number of timesteps.
        If not provided, we assume we sample once per timestep so, rate=dt

    - srate (float):
        This is related to rate, as it is the heat flux sampling rate in units of simulation time.
        This does not need to be provided if *rate* is already provided. Defaults are based on
        *rate* and *dt*. Units of [ns]

    - tau (int):
        max lag time to integrate over. This is in units of [ps]

    Args:
        **kwargs (dict):
            List of args above

    Returns:
        out (dict):\n
        - kx (ndarray): x-direction thermal conductivity [W/m/K]
        - ky (ndarray): y-direction thermal conductivity [W/m/K]
        - kz (ndarray): z-direction thermal conductivity [W/m/K]
        - t (ndarra): time [ps]
        - directory (str): directory of results
        - log (str): name of log file
        - dt (float): timestep [ps]
        - tot_time (float): total simulated time [ps]
        - tau (int): Lag time [ps]
        - T (float): [K]
        - vol (float): Volume of simulation cell  [angstroms^3]
        - srate (float): See above
        - jxjx (ndarray): x-direction heat flux autocorrelation
        - jyjy (ndarray): y-direction heat flux autocorrelation
        - jzjz (ndarray): z-direction heat flux autocorrelation

    '''

    original_dir = os.getcwd()

    try:

        # Check that directory exists
        directory = kwargs['directory'] if 'directory' in kwargs.keys() else './'
        if not os.path.isdir(directory):
            raise IOError('The path: {} is not a directory.'.format(directory))

        # go to the directory

        os.chdir(directory)

        # get heat flux, pass args
        hf = get_heat_flux(**kwargs)
        Jx = np.squeeze(hf['Jx'])
        Jy = np.squeeze(hf['Jy'])
        Jz = np.squeeze(hf['Jz'])

        T = kwargs['T'] if 'T' in kwargs.keys() else 300.
        vol = kwargs['vol'] if 'vol' in kwargs.keys() else 1.
        scale = __metal_to_SI(vol, T)

        Jx = Jx/vol
        Jy = Jy/vol
        Jz = Jz/vol

        log = kwargs['log'] if 'log' in kwargs.keys() else 'log.txt'
        # Set timestep
        if 'dt' in kwargs.keys():
            # If user passed value
            dt = kwargs['dt'] #[ps]
        else:
            # If not user passed value, try to find in log
            dts = extract_dt(log)
            dt = 1.0e-3 if len(dts) == 0 else dts[0]  #[ps]

        # set the heat flux sampling rate: rate*timestep*scaling
        if 'srate' in kwargs.keys():
            srate = kwargs['srate']
        else:
            if 'rate' in kwargs.keys():
                rate = kwargs['rate']
            elif 'rate' in hf.keys():
                rate = int(hf['rate'])
            else:
                rate = 1
            srate = rate*dt*1.0e-3 # [ns]

        # Calculate total time
        tot_time = srate*(len(Jx)-1)

        # set the integration limit (i.e. tau)
        tau = kwargs['tau'] if 'tau' in kwargs.keys() else tot_time # [ps]

        max_lag = int(floor(tau/(srate*1000.)))
        t = np.squeeze(np.linspace(0, (max_lag)*srate, max_lag+1))

        ### AUTOCORRELATION ###
        jxjx = autocorr(np.squeeze(Jx).astype(np.complex128), max_lag)
        jyjy = autocorr(np.squeeze(Jy).astype(np.complex128), max_lag)
        jzjz = autocorr(np.squeeze(Jz).astype(np.complex128), max_lag)

        ### INTEGRATION ###
        kx = integrate.cumtrapz(jxjx, t, initial=0)*scale
        ky = integrate.cumtrapz(jyjy, t, initial=0)*scale
        kz = integrate.cumtrapz(jzjz, t, initial=0)*scale

        # Return from directory
        os.chdir(original_dir)

        return {'kx':kx, 'ky':ky, 'kz':kz, 't':t, 'directory':directory,
                'log':log, 'dt':dt, 'tot_time':tot_time,'tau':tau, 'T':T,
                'vol':vol, 'srate':srate, 'jxjx':jxjx, 'jyjy':jyjy, 'jzjz':jzjz}

    except Exception as e:
        os.chdir(original_dir)
        print(traceback.format_exc())

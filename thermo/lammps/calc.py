
import numpy as np
import os
import sys
from scipy import integrate
from math import floor
import scipy.io as sio

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def __metal_to_SI( vol, T ):
    '''
    Converts LAMMPS metal units to SI units for thermal conductivity calculations.

    Args:
        vol (float):
            Volume in angstroms^3

        T (float):
            Temperature in K

    Returns:
        float: Converted value
    '''
    kb = 1.38064852e-23 #m^3*kg/(s^2*K)
    vol = vol/(1.0e10)**3 #to m^3
    #eV^2*ns/(ps^2*angstrom^4) to J^2/(s*m^4)
    to_SI = (1.602e-19)**2.*1.0e12*(1.0e10)**4.0*1000.
    return vol*to_SI/(kb*T**2)

def get_heat_flux(directory='.', heatflux_file='heat_out.heatflux',
                  mat_file='heat_flux.mat'):
    '''
    Gets the heat flux from a LAMMPS EMD simulation. Creates a compressed .mat
    file if only in text form. Loads .mat form if exists.

    Args:
        directory (str): This is the directory in which the simulation results
            are located. If not provided, the current directory is used.

        heatflux_file (str): Filename of heatflux output. If not provided
            'heat_out.heatflux' is used.

        mat_file (str): MATLAB file to load, if exists. If not provided,
            'heat_flux.mat' will be used. Also used as filename for saved MATLAB
            file.

    Returns:
        dict:Jx (list), Jy (list), Jz (list), rate (float)
    '''
    heatflux_file = os.path.join(directory, heatflux_file)
    mat_file = os.path.join(directory, mat_file)

    # Check that directory exists
    if not os.path.isdir(directory):
        raise IOError('The path: {} is not a directory.'.format(directory))

    # Go to directory and see if imported .mat file already exists
    if os.path.isfile(mat_file) and mat_file.endswith('.mat'):
        return sio.loadmat(mat_file)

    # Continue with the import since .mat file
    if not os.path.isfile(heatflux_file):
        raise IOError('The file: \'{}{}\' is not found.'.format(directory,heatflux_file))

    # Read the file
    with open(heatflux_file, 'r') as hf_file:
        lines = hf_file.readlines()[2:]

    num_elem = len(lines)-2

    # Get timestep
    rate = int(lines[0].split()[0])

    # read all data
    jx = np.zeros(num_elem)
    jy = np.zeros(num_elem)
    jz = np.zeros(num_elem)
    for i,line in enumerate(lines):
        vals = line.split()
        jx[i] = float(vals[1])
        jy[i] = float(vals[2])
        jz[i] = float(vals[3])

    output = {'Jx':jx, 'Jy':jy, 'Jz':jz, 'rate':rate}
    sio.savemat(mat_file, output)
    return output

def get_GKTC(directory='.', T=300, vol=1, dt=None, rate=None, tau=None,
             heatflux_file='heat_out.heatflux',mat_file='heat_flux.mat'):
    '''
    Gets the thermal conductivity vs. time profile using the Green-Kubo formalism.
    thermal conductivity vector and time vector.
    Assumptions with no info given by user:
    dt = 1 fs, vol = 1, T=300, rate=dt, tau=total time

    Args:
        directory (string):
            This is the directory in which the simulation results are located.
            If not provided, the current directory is used.

        T (float):
            This is the temperature at which the equlibrium simulation was run at.
            If not provided, T=300 is used. Units are in [K]

        vol (float):
            This is the volume of the simulation system.
            If not provided, vol=1 is used. Units are [angstroms^3].

        dt (float):
            This is the timestep of the green-kubo part of the simulation.
            If not provided, dt=1 fs is used. units are in [fs]

        rate (int):
            This is the rate at which the heat flux is sampled. This is in
            number of timesteps. If not provided, we assume we sample once per
            timestep so, rate=dt

        tau (int):
            max lag time to integrate over. This is in units of [ns]

        heatflux_file (str): Filename of heatflux output. If not provided
            'heat_out.heatflux' is used.

        mat_file (str): MATLAB file to load, if exists. If not provided,
            'heat_flux.mat' will be used. Also used as filename for saved MATLAB
            file.

    Returns:
        dict: kx, ky, kz, t, directory, dt, tot_time, tau, T, vol, srate,
        jxjx, jyjy, jzjz

    Output keys:\n
    - kx (ndarray): x-direction thermal conductivity [W/m/K]
    - ky (ndarray): y-direction thermal conductivity [W/m/K]
    - kz (ndarray): z-direction thermal conductivity [W/m/K]
    - t (ndarray): time [ns]
    - directory (str): directory of results
    - dt (float): timestep [fs]
    - tot_time (float): total simulated time [ns]
    - tau (int): Lag time [ns]
    - T (float): [K]
    - vol (float): Volume of simulation cell  [angstroms^3]
    - srate (float): See above
    - jxjx (ndarray): x-direction heat flux autocorrelation
    - jyjy (ndarray): y-direction heat flux autocorrelation
    - jzjz (ndarray): z-direction heat flux autocorrelation
    '''
    # Check that directory exists
    if not os.path.isdir(directory):
        raise IOError('The path: {} is not a directory.'.format(directory))

    # get heat flux, pass args
    hf = get_heat_flux(directory, heatflux_file,mat_file)
    Jx = np.squeeze(hf['Jx'])
    Jy = np.squeeze(hf['Jy'])
    Jz = np.squeeze(hf['Jz'])

    scale = __metal_to_SI(vol, T)

    # Set timestep if not set
    if dt is None:
        dt = 1.0e-6 # [ns]
    else:
        dt = dt*1.0e-6 # [fs] -> [ns]

    # set the heat flux sampling rate: rate*timestep*scaling
    srate = rate*dt*1.0e-6 # [ns]

    # Calculate total time
    tot_time = srate*(len(Jx)-1) # [ns]

    # set the integration limit (i.e. tau)
    if tau is None:
        tau = tot_time # [ns]

    max_lag = int(floor(tau/(srate)))
    t = np.squeeze(np.linspace(0, (max_lag)*srate, max_lag+1)) # [ns]

    ### AUTOCORRELATION ###
    jxjx = autocorr(np.squeeze(Jx).astype(np.complex128), max_lag)
    jyjy = autocorr(np.squeeze(Jy).astype(np.complex128), max_lag)
    jzjz = autocorr(np.squeeze(Jz).astype(np.complex128), max_lag)

    ### INTEGRATION ###
    kx = integrate.cumtrapz(jxjx, t, initial=0)*scale
    ky = integrate.cumtrapz(jyjy, t, initial=0)*scale
    kz = integrate.cumtrapz(jzjz, t, initial=0)*scale

    dt/=1e6 # [ns] -> [fs]

    return {'kx':kx, 'ky':ky, 'kz':kz, 't':t, 'directory':directory,
            'dt':dt, 'tot_time':tot_time,'tau':tau, 'T':T,
            'vol':vol, 'srate':srate, 'jxjx':jxjx, 'jyjy':jyjy, 'jzjz':jzjz}

import numpy as np
import os
from scipy import integrate
from math import floor
import scipy.io as sio
from thermo.math.correlate import autocorr

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def __metal_to_SI(vol, T):
    """
    Converts LAMMPS metal units to SI units for thermal conductivity calculations.

    Args:
        vol (float):
            Volume in angstroms^3

        T (float):
            Temperature in K

    Returns:
        float: Converted value
    """
    kb = 1.38064852e-23  # m3*kg/(s2*K)
    vol = vol/(1.0e10)**3  # to m3
    # eV2*ns/(ps2*A4) to J2/(s*m4)
    to_SI = (1.602e-19)**2.*1.0e12*(1.0e10)**4.0*1000.
    return vol*to_SI/(kb*T**2)


def get_heat_flux(directory='.', heatflux_file='heat_out.heatflux', mat_file='heat_flux.mat'):
    """
    Gets the heat flux from a LAMMPS EMD simulation. Creates a compressed .mat
    file if only in text form. Loads .mat form if exists.

    Args:
        directory (str):
            Directory of simulation results

        heatflux_file (str):
            Filename of heatflux output

        mat_file (str):
            MATLAB file to load, if exists, or save to, if does not exist.
            Default save name of 'heat_flux.mat'

    Returns:
        dict: Dictionary with heat flux data

    .. csv-table:: Output dictionary (metal units)
       :stub-columns: 1

       **key**,jx,jy,jz,rate
       **units**,|j1|,|j1|,|j1|,timestep

    .. |j1| replace:: eV ps\ :sup:`-1` A\ :sup:`-2`
    """
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

    num_elem = len(lines)

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

    output = {'jx':jx, 'jy':jy, 'jz':jz, 'rate':rate}
    sio.savemat(mat_file, output)
    return output


def get_GKTC(directory='.', T=300, vol=1, dt=None, rate=None, tau=None,
             heatflux_file='heat_out.heatflux',mat_file='heat_flux.mat'):
    """
    Calculates the thermal conductivity (TC) using the Green-Kubo (GK) formalism.
    The 'metal' units in LAMMPS must be used.

    Args:
        directory (string):
            Directory of simulation

        T (float):
            Temperature of simulation. Units of K

        vol (float):
            Volume of the simulation cell. Units of A^3

        dt (float):
            Timestep of the of simulation. Units are fs

        rate (int):
            Rate at which the heat flux is sampled in number of timesteps. Default of rate=dt

        tau (int):
            max lag time to integrate over. Units of ns and default of tau=total time

        heatflux_file (str):
            Heatflux output filename.

        mat_file (str):
            MATLAB file to load, if exists, or save to, if does not exist.
            Default save name of 'heat_flux.mat'

    Returns:
        dict: Dictionary with Green-Kubo thermal conductivity data

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,kx,ky,kz,t,dt,T,V,jxjx,jyjy,jzjz,tot_time,tau,srate,directory
       **units**,|gk1|,|gk1|,|gk1|,ns,fs,K,|gk2|,|gk3|,|gk3|,|gk3|,ns,ns,ns,N/A

    .. |gk1| replace:: Wm\ :sup:`-1` K\ :sup:`-1`
    .. |gk2| replace:: A\ :sup:`3`
    .. |gk3| replace:: (eV ps\ :sup:`-1` A\ :sup:`-2`)\ :sup:`2`
    """
    # Check that directory exists
    if not os.path.isdir(directory):
        raise IOError('The path: {} is not a directory.'.format(directory))

    # get heat flux, pass args
    hf = get_heat_flux(directory, heatflux_file,mat_file)
    jx = np.squeeze(hf['jx'])
    jy = np.squeeze(hf['jy'])
    jz = np.squeeze(hf['jz'])

    scale = __metal_to_SI(vol, T)

    # Set timestep if not set
    if dt is None:
        dt = 1.0e-6  # [ns]
    else:
        dt = dt*1.0e-6  # [fs] -> [ns]

    # set the heat flux sampling rate: rate*timestep*scaling
    srate = rate*dt  # [ns]

    # Calculate total time
    tot_time = srate*(len(jx)-1)  # [ns]

    # set the integration limit (i.e. tau)
    if tau is None:
        tau = tot_time  # [ns]

    max_lag = int(floor(tau/(srate)))
    t = np.squeeze(np.linspace(0, (max_lag)*srate, max_lag+1))  # [ns]

    jxjx = autocorr(np.squeeze(jx).astype(np.complex128), max_lag)
    jyjy = autocorr(np.squeeze(jy).astype(np.complex128), max_lag)
    jzjz = autocorr(np.squeeze(jz).astype(np.complex128), max_lag)

    kx = integrate.cumtrapz(jxjx, t, initial=0)*scale
    ky = integrate.cumtrapz(jyjy, t, initial=0)*scale
    kz = integrate.cumtrapz(jzjz, t, initial=0)*scale

    dt /= 1e6  # [ns] -> [fs]

    return {'kx':kx, 'ky':ky, 'kz':kz, 't':t, 'directory':directory,
            'dt':dt, 'tot_time':tot_time,'tau':tau, 'T':T,
            'V':vol, 'srate':srate, 'jxjx':jxjx, 'jyjy':jyjy, 'jzjz':jzjz}

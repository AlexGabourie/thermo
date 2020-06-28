import numpy as np
from scipy.integrate import cumtrapz
from .data import load_shc
from .data import __get_direction  # TODO move function to more accessible location
from thermo.math.correlate import corr
from scipy import integrate
import os

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def __scale_gpumd_tc(vol, T):
    """
    Used to scale the thermal conductivity when converting GPUMD heat-flux correlations
    to thermal conductivity.

    Args:
        vol (float):
            Volume in angstroms^3

        T (float):
            Temperature in K

    Returns:
        float: Converted value
    """

    one = 1.602176634e-19 * 9.651599e7  # eV^3/amu -> Jm^2/s^2*eV
    two = 1. / 1.e15  # fs -> s
    three = 1.e30 / 8.617333262145e-5  # K/(eV*Ang^3) -> K/(eV*m^3) w/ Boltzmann
    return one * two * three / (T * T * vol)


def get_gkma_kappa(data, nbins, nsamples, dt, sample_interval, T=300, vol=1, max_tau=None, directions='xyz',
                   outputfile='heatmode.npy', save=False, directory=None, return_data=True):
    """
    Calculate the Green-Kubo thermal conductivity from modal heat current data from 'load_heatmode'

    Args:
        data (dict):
            Dictionary with heat currents loaded by 'load_heatmode'

        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times heat flux was sampled with GKMA during GPUMD simulation

        dt (float):
            Time step during data collection in fs

        sample_interval (int):
            Number of time steps per sample of modal heat flux

        T (float):
            Temperature of system during data collection

        vol (float):
            Volume of system in angstroms^3

        max_tau (float):
            Correlation time to calculate up to. Units of ns

        directions (str):
            Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed (i.e. 'xz'
            is accepted)

        outputfile (str):
            File name to save read data to. Output file is a binary dictionary. Loading from a binary file is much
            faster than re-reading data files and saving is recommended

        save (bool):
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended (default:
            False)

        directory (str):
            Name of directory storing the input file to read

        return_data (bool):
            Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True (default: True)

    Returns:
        dict: Input data dict but with correlation, thermal conductivity, and lag time data included

    """

    if not directory:
        out_path = os.path.join(os.getcwd(), outputfile)
    else:
        out_path = os.path.join(directory, outputfile)

    scale = __scale_gpumd_tc(vol, T)
    # set the heat flux sampling time: rate * timestep * scaling
    srate = sample_interval * dt  # [fs]

    # Calculate total time
    tot_time = srate * (nsamples - 1)  # [fs]

    # set the integration limit (i.e. tau)
    if max_tau is None:
        max_tau = tot_time  # [fs]
    else:
        max_tau = max_tau * 1e6  # [fs]

    max_lag = int(np.floor(max_tau / srate))
    size = max_lag + 1
    data['tau'] = np.squeeze(np.linspace(0, max_lag * srate, max_lag + 1))  # [ns]

    ### AUTOCORRELATION ###
    directions = __get_direction(directions)
    cplx = np.complex128
    # Note: loops necessary due to memory constraints
    #  (can easily max out cluster mem.)
    if 'x' in directions:
        if 'jmxi' not in data.keys() or 'jmxo' not in data.keys():
            raise ValueError("x direction data is missing")

        jxi = np.sum(data['jmxi'], axis=0)
        data['corr_xmi_xi'] = np.zeros((nbins, size))
        data['kmxi'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['corr_xmi_xi'][m, :] = corr(data['jmxi'][m, :].astype(cplx), jxi.astype(cplx), max_lag)
            data['kmxi'][m, :] = integrate.cumtrapz(data['corr_xmi_xi'][m, :], data['tau'], initial=0) * scale
        del jxi

        jxo = np.sum(data['jmxo'], axis=0)
        data['corr_xmo_xo'] = np.zeros((nbins, size))
        data['kmxo'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['corr_xmo_xo'][m, :] = corr(data['jmxo'][m, :].astype(cplx), jxo.astype(cplx), max_lag)
            data['kmxo'][m, :] = integrate.cumtrapz(data['corr_xmo_xo'][m, :], data['tau'], initial=0) * scale
        del jxo

    if 'y' in directions:
        if 'jmyi' not in data.keys() or 'jmyo' not in data.keys():
            raise ValueError("y direction data is missing")

        jyi = np.sum(data['jmyi'], axis=0)
        data['corr_ymi_yi'] = np.zeros((nbins, size))
        data['kmyi'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['corr_ymi_yi'][m, :] = corr(data['jmyi'][m, :].astype(cplx), jyi.astype(cplx), max_lag)
            data['kmyi'][m, :] = integrate.cumtrapz(data['corr_ymi_yi'][m, :], data['tau'], initial=0) * scale
        del jyi

        jyo = np.sum(data['jmyo'], axis=0)
        data['corr_ymo_yo'] = np.zeros((nbins, size))
        data['kmyo'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['corr_ymo_yo'][m, :] = corr(data['jmyo'][m, :].astype(cplx), jyo.astype(cplx), max_lag)
            data['kmyo'][m, :] = integrate.cumtrapz(data['corr_ymo_yo'][m, :], data['tau'], initial=0) * scale
        del jyo

    if 'z' in directions:
        if 'jmz' not in data.keys():
            raise ValueError("z direction data is missing")

        jz = np.sum(data['jmz'], axis=0)
        data['corr_zm_z'] = np.zeros((nbins, size))
        data['kmz'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['corr_zm_z'][m, :] = corr(data['jmz'][m, :].astype(cplx), jz.astype(cplx), max_lag)
            data['kmz'][m, :] = integrate.cumtrapz(data['corr_zm_z'][m, :], data['tau'], initial=0) * scale
        del jz

    data['tau'] = data['tau'] / 1.e6

    if save:
        np.save(out_path, data)

    if return_data:
        return data
    return


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

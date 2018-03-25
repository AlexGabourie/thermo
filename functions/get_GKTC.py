import os
import sys
import numpy as np
from scipy import integrate
from math import floor
from autocorr import autocorr
from get_heat_flux import get_heat_flux
from extract_dt import extract_dt
from metalToSI_TC import metalToSI_TC
import traceback


def get_GKTC(**kwargs):
    '''Gets the thermal conductivity vs. time profile using the Green-Kubo formalism. 
    thermal conductivity vector and time vector.
    Assumptions with no info given by user:
    dt = 1 fs, vol = 1, T=300, rate=dt, tau=tot_time'''
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
        scale = metalToSI_TC(vol, T)

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
        print(np.shape(t))
        print(np.shape(jxjx))
        print(rate)
        
        ### INTEGRATION ###
        kx = integrate.cumtrapz(jxjx, t, initial=0)*scale
        ky = integrate.cumtrapz(jyjy, t, initial=0)*scale
        kz = integrate.cumtrapz(jzjz, t, initial=0)*scale
        
        # Return from directory
        os.chdir(original_dir)
        
        return {'kx':kx, 'ky':ky, 'kz':kz, 't':t, 'directory':directory, 'log':log, 'dt':dt, 'tot_time':tot_time, 
                    'tau':tau, 'T':T, 'vol':vol, 'srate':srate, 'jxjx':jxjx, 'jyjy':jyjy, 'jzjz':jzjz}
    
    except Exception as e:
        os.chdir(original_dir)
        print(traceback.format_exc())
    

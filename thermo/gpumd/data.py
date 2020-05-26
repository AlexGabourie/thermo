import numpy as np
import os

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

#########################################
# Data-loading Related
#########################################

def load_sdc(Nc, num_run=1, average=False, directory='', filename='sdc.out'):
    """
    Loads data from sdc.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Number of time correlation points the VAC/SDC is computed for. For num_run>1,
            a list can be provided to specify number of points in each run if they
            are different. Otherwise, it is assumed that the same number of points
            are used per run

        num_run (int):
            Number of SDC runs in the sdc.out file

        average (bool):
            Averages all of the runs to a single output. Default is False. Only works
            if points_per_run is an int.

        directory (str):
            Directory to load 'sdc.out' file from (dir. of simulation)

        filename (str):
            File to load SDC from. Default is sdc.out

    Returns:
        tuple: sdc, vac

    sdc (dict(dict)):
    Dictionary with SDC data. The outermost dictionary stores each individual run.
    Each run is a dictionary with keys:\n
    - t (ps)
    - SDC_x (Angstrom^2/ps)
    - SDC_y (Angstrom^2/ps)
    - SDC_z (Angstrom^2/ps)

    If average=True, this will also be stored as a run with the same run keys.

    vac (dict(dict)):
    Dictonary with VAC data. The outermost dictionary stores each individual run.
    Each run is a dictionary with keys:\n
    - t (ps)
    - VAC_x (Angstrom^2/ps^2)
    - VAC_y (Angstrom^2/ps^2)
    - VAC_z (Angstrom^2/ps^2)

    If average=True, this will also be stored as a run with the same run keys.

    """
    is_int = type(Nc) == int
    # do input checks
    if ( not is_int and average):
        raise ValueError('average cannot be used if points_per_run is not an int.')

    if (not is_int and len(Nc) != num_run):
        raise ValueError('length of points_per_run must be equal to num_run.')

    if (not is_int and len(Nc) == 1):
        points_per_run = points_per_run[0]

    if directory=='':
        sdc_path = os.path.join(os.getcwd(), filename)
    else:
        sdc_path = os.path.join(directory, filename)

    with open(sdc_path, 'r') as f:
        lines = f.readlines()

    vac = dict()
    sdc = dict()
    idx_shift = 0
    for run_num in range(num_run):
        if is_int:
            pt_rng = Nc
        else:
            pt_rng = Nc[run_num]

        vrun = dict()
        srun = dict()
        vrun['t'] = np.zeros((pt_rng))
        srun['t'] = np.zeros((pt_rng))

        vrun['VAC_x'] = np.zeros((pt_rng))
        vrun['VAC_y'] = np.zeros((pt_rng))
        vrun['VAC_z'] = np.zeros((pt_rng))

        srun['SDC_x'] = np.zeros((pt_rng))
        srun['SDC_y'] = np.zeros((pt_rng))
        srun['SDC_z'] = np.zeros((pt_rng))
        for point in range(pt_rng):
            data = lines[idx_shift + point].split()
            srun['t'][point] = float(data[0])
            vrun['t'][point] = float(data[0])

            vrun['VAC_x'][point] = float(data[1])
            vrun['VAC_y'][point] = float(data[2])
            vrun['VAC_z'][point] = float(data[3])

            srun['SDC_x'][point] = float(data[4])
            srun['SDC_y'][point] = float(data[5])
            srun['SDC_z'][point] = float(data[6])
        idx_shift += pt_rng

        vac['run'+str(run_num)] = vrun
        sdc['run'+str(run_num)] = srun

    if average:
        vave = dict()
        save = dict()

        vave['t'] = np.zeros((pt_rng))
        vave['VAC_x'] = np.zeros((pt_rng))
        vave['VAC_y'] = np.zeros((pt_rng))
        vave['VAC_z'] = np.zeros((pt_rng))

        save['t'] = np.zeros((pt_rng))
        save['SDC_x'] = np.zeros((pt_rng))
        save['SDC_y'] = np.zeros((pt_rng))
        save['SDC_z'] = np.zeros((pt_rng))

        for key in sdc:
            vrun = vac[key]
            srun = sdc[key]

            vave['t'] += vrun['t']
            vave['VAC_x'] += vrun['VAC_x']
            vave['VAC_y'] += vrun['VAC_y']
            vave['VAC_z'] += vrun['VAC_z']

            save['t'] += srun['t']
            save['SDC_x'] += srun['SDC_x']
            save['SDC_y'] += srun['SDC_y']
            save['SDC_z'] += srun['SDC_z']

        vave['t'] /= num_run
        vave['VAC_x'] /= num_run
        vave['VAC_y'] /= num_run
        vave['VAC_z'] /= num_run

        save['t'] /= num_run
        save['SDC_x'] /= num_run
        save['SDC_y'] /= num_run
        save['SDC_z'] /= num_run

        sdc['ave'] = save
        vac['ave'] = vave

    return sdc, vac



def load_vac(Nc, num_run=1, average=False, directory='', filename='mvac.out'):
    """
    Loads data from mvac.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Number of time correlation points the VAC is computed for. For num_run>1,
            a list can be provided to specify number of points in each run if they
            are different. Otherwise, it is assumed that the same number of points
            are used per run

        num_run (int):
            Number of VAC runs in the mvac.out file

        average (bool):
            Averages all of the runs to a single output. Default is False. Only works
            if points_per_run is an int.

        directory (str):
            Directory to load 'mvac.out' file from (dir. of simulation)

        filename (str):
            File to load VAC from. Default is mvac.out

    Returns:
        dict(dict):
            Dictonary with VAC data. The outermost dictionary stores each individual run.

    Each run is a dictionary with keys:\n
    - t (ps)
    - VAC_x (Angstrom^2/ps^2)
    - VAC_y (Angstrom^2/ps^2)
    - VAC_z (Angstrom^2/ps^2)

    If average=True, this will also be stored as a run with the same run keys.
    """
    is_int = type(Nc) == int
    # do input checks
    if ( not is_int and average):
        raise ValueError('average cannot be used if points_per_run is not an int.')

    if (not is_int and len(Nc) != num_run):
        raise ValueError('length of points_per_run must be equal to num_run.')

    if (not is_int and len(Nc) == 1):
        points_per_run = points_per_run[0]

    if directory=='':
        vac_path = os.path.join(os.getcwd(), filename)
    else:
        vac_path = os.path.join(directory, filename)

    with open(vac_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    idx_shift = 0
    for run_num in range(num_run):
        if is_int:
            pt_rng = Nc
        else:
            pt_rng = Nc[run_num]

        run = dict()
        run['t'] = np.zeros((pt_rng))
        run['VAC_x'] = np.zeros((pt_rng))
        run['VAC_y'] = np.zeros((pt_rng))
        run['VAC_z'] = np.zeros((pt_rng))
        for point in range(pt_rng):
            data = lines[idx_shift + point].split()
            run['t'][point] = float(data[0])
            run['VAC_x'][point] = float(data[1])
            run['VAC_y'][point] = float(data[2])
            run['VAC_z'][point] = float(data[3])
        idx_shift += pt_rng

        out['run'+str(run_num)] = run

    if average:
        ave = dict()
        ave['t'] = np.zeros((pt_rng))
        ave['VAC_x'] = np.zeros((pt_rng))
        ave['VAC_y'] = np.zeros((pt_rng))
        ave['VAC_z'] = np.zeros((pt_rng))

        for key in out:
            run = out[key]
            ave['t'] += run['t']
            ave['VAC_x'] += run['VAC_x']
            ave['VAC_y'] += run['VAC_y']
            ave['VAC_z'] += run['VAC_z']

        ave['t'] /= num_run
        ave['VAC_x'] /= num_run
        ave['VAC_y'] /= num_run
        ave['VAC_z'] /= num_run

        out['ave'] = ave

    return out


def load_dos(points_per_run, num_run=1, average=False, directory='', filename='dos.out'):
    """
    Loads data from dos.out GPUMD output file.

    Args:
        points_per_run (int or list(int)):
            Number of frequency points the DOS is computed for. For num_run>1,
            a list can be provided to specify number of points in each run if they
            are different. Otherwise, it is assumed that the same number of points
            are used per run

        num_run (int):
            Number of DOS runs in the dos.out file

        average (bool):
            Averages all of the runs to a single output. Default is False. Only works
            if points_per_run is an int.

        directory (str):
            Directory to load 'dos.out' file from (dir. of simulation)

        filename (str):
            File to load DOS from. Default is dos.out

    Returns:
        dict(dict)): Dictonary with DOS data. The outermost dictionary stores
        each individual run.

    Each run is a dictionary with keys:\n
    - nu (THz)
    - DOS_x (1/THz)
    - DOS_y (1/THz)
    - DOS_z (1/THz)

    If average=True, this will also be stored as a run with the same run keys.
    """
    is_int = type(points_per_run) == int
    # do input checks
    if ( not is_int and average):
        raise ValueError('average cannot be used if points_per_run is not an int.')

    if (not is_int and len(points_per_run) != num_run):
        raise ValueError('length of points_per_run must be equal to num_run.')

    if (not is_int and len(points_per_run) == 1):
        points_per_run = points_per_run[0]

    if directory=='':
        dos_path = os.path.join(os.getcwd(), filename)
    else:
        dos_path = os.path.join(directory, filename)

    with open(dos_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    idx_shift = 0
    for run_num in range(num_run):
        if is_int:
            pt_rng = points_per_run
        else:
            pt_rng = points_per_run[run_num]

        run = dict()
        run['nu'] = np.zeros((pt_rng))
        run['DOS_x'] = np.zeros((pt_rng))
        run['DOS_y'] = np.zeros((pt_rng))
        run['DOS_z'] = np.zeros((pt_rng))
        for point in range(pt_rng):
            data = lines[idx_shift + point].split()
            run['nu'][point] = float(data[0])/(6.283185307179586)
            run['DOS_x'][point] = float(data[1])
            run['DOS_y'][point] = float(data[2])
            run['DOS_z'][point] = float(data[3])
        idx_shift += pt_rng

        out['run'+str(run_num)] = run

    if average:
        ave = dict()
        ave['nu'] = np.zeros((pt_rng))
        ave['DOS_x'] = np.zeros((pt_rng))
        ave['DOS_y'] = np.zeros((pt_rng))
        ave['DOS_z'] = np.zeros((pt_rng))

        for key in out:
            run = out[key]
            ave['nu'] += run['nu']
            ave['DOS_x'] += run['DOS_x']
            ave['DOS_y'] += run['DOS_y']
            ave['DOS_z'] += run['DOS_z']

        ave['nu'] /= num_run
        ave['DOS_x'] /= num_run
        ave['DOS_y'] /= num_run
        ave['DOS_z'] /= num_run

        out['ave'] = ave

    return out


def load_shc(Nc, directory='', filename='shc.out'):
    """
    Loads the data from shc.out GPUMD output file.

    Args:
        Nc (int):
            Maximum number of correlation steps

        directory (str):
            Directory to load 'shc.out' file from (dir. of simulation)

        filename (str):
            File to load SHC from. Default is shc.out

    Returns:
        dict: Dictionary of in- and out-of-plane shc results (average)
    """
    if (not type(Nc) == int):
        raise ValueError('Nc must be an int.')
        
    if directory=='':
        shc_path = os.path.join(os.getcwd(),filename)
    else:
        shc_path = os.path.join(directory,filename)

    with open(shc_path, 'r') as f:
        lines = f.readlines()

    shc = np.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        data = line.split()
        shc[i, 0] = float(data[0])
        shc[i, 1] = float(data[1])

    Ns = shc.shape[0]//Nc
    shc_in = np.reshape(shc[:,0], (Ns, Nc))
    shc_out = np.reshape(shc[:,1], (Ns, Nc))
    shc_in = np.mean(shc_in,0)*1000./10.18 # eV/ps
    shc_out = np.mean(shc_out,0)*1000./10.18

    out = dict()
    out['shc_in'] = shc_in
    out['shc_out'] = shc_out
    return out

def load_kappa(directory='', filename='kappa.out'):
    """
    Loads data from kappa.out GPUMD output file which contains HNEMD kappa.

    Args:
        directory (str):
            Directory to load 'kappa.out' file from (dir. of simulation)

        filename (str):
            File to load kappa from. Default is kappa.out

    Returns:
        dict: A dictionary with keys corresponding to the columns in 'kappa.out'
    """

    if directory=='':
        kappa_path = os.path.join(os.getcwd(),filename)
    else:
        kappa_path = os.path.join(directory,filename)

    with open(kappa_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    out['kx_in'] = np.zeros(len(lines))
    out['kx_out'] = np.zeros(len(lines))
    out['ky_in'] = np.zeros(len(lines))
    out['ky_out'] = np.zeros(len(lines))
    out['kz'] = np.zeros(len(lines))

    for i, line in enumerate(lines):
        nums = line.split()
        out['kx_in'][i] = float(nums[0])
        out['kx_out'][i] = float(nums[1])
        out['ky_in'][i] = float(nums[2])
        out['ky_out'][i] = float(nums[3])
        out['kz'][i] = float(nums[4])

    return out

def load_hac(directory='',filename='hac.out'):
    """
    Loads data from hac.out GPUMD output file which contains the
    heat-current autocorrelation and running thermal conductivity values.

    **Created for GPUMD-v1.9**

    Args:
        directory (str): Directory storing heat flux file.
        filename (str): File to load hac from. Default is 'hac.out'

    Returns:
        dict: A dictionary with keys corresponding to the columns in
        'hac.out' with some additional keys for aggregated values (see description)

    Units: hacf [ev^3/amu]; k [W/m/K]; t [ps]

    Abbreviated description of keys in output:\n
    * hacf_x: ave. of i/o components
    * hacf_y: ave. of i/o components
    * k_x: ave. of i/o components
    * k_y: ave. of i/o components
    * k_i: ave. of x/y components
    * k_o: ave. of x/y components
    * k: ave of all in-plane components
    * t: correlation time
    """

    if directory=='':
        hac_path = os.path.join(os.getcwd(),filename)
    else:
        hac_path = os.path.join(directory,filename)

    with open(hac_path, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        t = np.zeros((N,1))
        x_ac_i = np.zeros((N,1)) # autocorrelation IN, X
        x_ac_o = np.zeros((N,1)) # autocorrelation OUT, X

        y_ac_i = np.zeros((N,1)) # autocorrelation IN, Y
        y_ac_o = np.zeros((N,1)) # autocorrelation OUT, Y

        z_ac = np.zeros((N,1)) # autocorrelation Z

        kx_i = np.zeros((N,1)) # kappa IN, X
        kx_o = np.zeros((N,1)) # kappa OUT, X

        ky_i = np.zeros((N,1)) # kappa IN, Y
        ky_o = np.zeros((N,1)) # kappa OUT, Y

        kz = np.zeros((N,1)) # kappa, Z

        for i, line in enumerate(lines):
            vals = line.split()
            t[i] = vals[0]
            x_ac_i[i] = vals[1]
            x_ac_o[i] = vals[2]
            y_ac_i[i] = vals[3]
            y_ac_o[i] = vals[4]
            z_ac[i] = vals[5]
            kx_i[i] = vals[6]
            kx_o[i] = vals[7]
            ky_i[i] = vals[8]
            ky_o[i] = vals[9]
            kz[i] = vals[10]

    out = dict()
    # x-direction heat flux autocorrelation function
    out['hacf_xi'] = x_ac_i
    out['hacf_xo'] = x_ac_o
    out['hacf_x'] = x_ac_i + x_ac_o

    # y-direction heat flux autocorrelation function
    out['hacf_yi'] = y_ac_i
    out['hacf_yo'] = y_ac_o
    out['hacf_y'] = y_ac_i + y_ac_o

    # z-direction heat flux autocorrelation function
    out['hacf_z'] = z_ac

    # x-direction thermal conductivity
    out['k_xi'] = kx_i
    out['k_xo'] = kx_o
    out['k_x'] = kx_i + kx_o

    # y-direction thermal conductivity
    out['k_yi'] = ky_i
    out['k_yo'] = ky_o
    out['k_y'] = ky_i + ky_o

    # z-direction thermal conductivity
    out['k_z'] = kz

    # Combined thermal conductivities (isotropic)
    out['k_i'] = (kx_i + ky_i)/2.
    out['k_o'] = (kx_o + ky_o)/2.
    out['k'] = (out['k_x'] + out['k_y'])/2.

    out['t'] = t

    return out

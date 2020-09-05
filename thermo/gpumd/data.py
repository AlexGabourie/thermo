import numpy as np
import pandas as pd
import os
import copy
import multiprocessing as mp
from functools import partial
from collections import deque
from .common import __get_path, __get_direction, __check_list, __check_range


__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

#########################################
# Helper Functions
#########################################


def __process_sample(nbins, i):
    """
    A helper function for the multiprocessing of kappamode.out files

    Args:
        nbins (int):
            Number of bins used in the GPUMD simulation

        i (int):
            The current sample from a run to analyze

    Returns:
        np.ndarray: A 2D array of each bin and output for a sample


    """
    out = list()
    for j in range(nbins):
        out += [float(x) for x in malines[j + i * nbins].split()]
    return np.array(out).reshape((nbins,5))


def tail(f, nlines, BLOCK_SIZE=32768):
    """
    Reads the last nlines of a file.

    Args:
        f (filehandle):
            File handle of file to be read

        nlines (int):
            Number of lines to be read from end of file

        BLOCK_SIZE (int):
            Size of block (in bytes) to be read per read operation.
            Performance depend on this parameter and file size.

    Returns:
        list: List of ordered final nlines of file

    Additional Information:
    Since GPUMD output files are mostly append-only, this becomes
    useful when a simulation prematurely ends (i.e. cluster preempts
    run, but simulation restarts elsewhere). In this case, it is not
    necessary to clean the directory before re-running. File outputs
    will be too long (so there still is a storage concern), but the
    proper data can be extracted from the end of file.
    This may also be useful if you want to only grab data from the
    final m number of runs of the simulation
    """
    # BLOCK_SIZE is in bytes (must decode to string)
    f.seek(0, 2)
    bytes_remaining = f.tell()
    idx = -BLOCK_SIZE
    blocks = list()
    # Make no assumptions about line length
    lines_left = nlines
    eof = False
    first = True
    num_lines = 0

    # BLOCK_size is smaller than file
    if BLOCK_SIZE <= bytes_remaining:
        while lines_left > 0 and not eof:
            if bytes_remaining > BLOCK_SIZE:
                f.seek(idx, 2)
                blocks.append(f.read(BLOCK_SIZE))
            else:  # if reached end of file
                f.seek(0, 0)
                blocks.append(f.read(bytes_remaining))
                eof = True

            idx -= BLOCK_SIZE
            bytes_remaining -= BLOCK_SIZE
            num_lines = blocks[-1].count(b'\n')
            if first:
                lines_left -= num_lines - 1
                first = False
            else:
                lines_left -= num_lines

            # since whitespace removed from eof, must compare to 1 here
            if eof and lines_left > 1:
                raise ValueError("More lines requested than exist.")

        # Corrects for reading too many lines with large buffer
        if bytes_remaining > 0:
            skip = 1 + abs(lines_left)
            blocks[-1] = blocks[-1].split(b'\n', skip)[skip]
        text = b''.join(reversed(blocks)).strip()
    else:  # BLOCK_SIZE is bigger than file
        f.seek(0, 0)
        block = f.read()
        num_lines = block.count(b'\n')
        if num_lines < nlines:
            raise ValueError("More lines requested than exist.")
        skip = num_lines - nlines
        text = block.split(b'\n', skip)[skip].strip()
    return text.split(b'\n')


def __modal_analysis_read(nbins, nsamples, datapath,
                          ndiv, multiprocessing, ncore, block_size):

    global malines
    # Get full set of results
    datalines = nbins * nsamples
    with open(datapath, 'rb') as f:
        if multiprocessing:
            malines = tail(f, datalines, BLOCK_SIZE=block_size)
        else:
            malines = deque(tail(f, datalines, BLOCK_SIZE=block_size))

    if multiprocessing:  # TODO Improve memory efficiency of multiprocessing
        if not ncore:
            ncore = mp.cpu_count()

        func = partial(__process_sample, nbins)
        pool = mp.Pool(ncore)
        data = np.array(pool.map(func, range(nsamples)), dtype='float32').transpose((1, 0, 2))
        pool.close()

    else:  # Faster if single thread
        data = np.zeros((nbins, nsamples, 5), dtype='float32')
        for j in range(nsamples):
            for i in range(nbins):
                measurements = malines.popleft().split()
                data[i, j, 0] = float(measurements[0])
                data[i, j, 1] = float(measurements[1])
                data[i, j, 2] = float(measurements[2])
                data[i, j, 3] = float(measurements[3])
                data[i, j, 4] = float(measurements[4])

    del malines
    if ndiv:
        nbins = int(np.ceil(data.shape[0] / ndiv))  # overwrite nbins
        npad = nbins * ndiv - data.shape[0]
        data = np.pad(data, [(0, npad), (0, 0), (0, 0)])
        data = np.sum(data.reshape((-1, ndiv, data.shape[1], data.shape[2])), axis=1)

    return data


#########################################
# Data-loading Related
#########################################

def load_compute(quantities=None, directory=None, filename='compute.out'):
    """
    loads data from compute.out GPUMD output file 

    Args:
        quantities (str or list(str)):
            Quantities to extract from compute.out Accepted quantities are:
            ['temperature', 'potential', 'force', 'virial', 'jp', 'jk']. Other quantity will be ignored.

        directory (str):
            Directory to load 'compute.out' file from (dir. of simulation)

        filename (str):
            file to load compute from

        Returns:
            'output' dictionary containing the data from compute.out
    """
    if not quantities:
        return None
    com_n = __get_path(directory, filename)
    com_n = pd.read_csv(com_n, sep="\s+", header=None)

    total_cols = len(com_n.columns)
    q_count = {'temperature': 1, 'potential': 1, 'force': 3, 'virial': 3, 'jp': 3, 'jk': 3}
    output = dict()

    count = 0
    for value in quantities:
        count += q_count[value]

    m = int(total_cols / count)
    if 'temperature' in quantities:
        m = int((total_cols - 2) / count)

    start = 0
    if 'temperature' in quantities:
        output['temperature'] = np.array(com_n.iloc[:, :m])
        output['heat_in'] = np.array(com_n.iloc[:, total_cols - 1:total_cols])
        output['heat_out'] = np.array(com_n.iloc[:, total_cols:])
        start = m

    if 'potential' in quantities:
        output['potential'] = np.array(com_n.iloc[:, start: m])
        start += m

    if 'force' in quantities:
        output['force'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start += 3 * m

    if 'virial' in quantities:
        output['virial'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start += 3 * m

    if 'jp' in quantities:
        output['jp'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start += 3 * m

    if 'jk' in quantities:
        output['jk'] = np.array(com_n.iloc[:, start: start + (3 * m)])

    return output


def load_thermo(directory=None, filename='thermo.out'):
    """
    Loads data from thermo.out GPUMD output file.

    Args:
        directory (str):
            Directory to load 'thermo.out' file from

        filename (str):
            Name of thermal data file

        Returns:
            'output' dictionary containing the data from thermo.out

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,T,K,U,Px,Py,Pz,Lx,Ly,Lz,ax,ay,az,bx,by,bz,cx,cy,cz
       **units**,K,eV,eV,GPa,GPa,GPa,A,A,A,A,A,A,A,A,A,A,A,A

    """
    thermo_path = __get_path(directory, filename)
    data = pd.read_csv(thermo_path, delim_whitespace=True, header=None)
    labels = ['T', 'K', 'U', 'Px', 'Py', 'Pz']
    # Orthogonal
    if data.shape[1] == 9:
        labels += ['Lx', 'Ly', 'Lz']
    elif data.shape[1] == 15:
        labels += ['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']

    out = dict()
    for i in range(data.shape[1]):
        out[labels[i]] = data[i].to_numpy()

    return out


def load_heatmode(nbins, nsamples, directory=None,
                  inputfile='heatmode.out', directions='xyz',
                  outputfile='heatmode.npy', ndiv=None, save=False,
                  multiprocessing=False, ncore=None, block_size=65536, return_data=True):
    """
    Loads data from heatmode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory usage may be significantly larger than file size

    Args:
        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times heat flux was sampled with GKMA during GPUMD simulation

        directory (str):
            Name of directory storing the input file to read

        inputfile (str):
            Modal heat flux file output by GPUMD

        directions (str):
            Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed (i.e. 'xz'
            is accepted)

        outputfile (str):
            File name to save read data to. Output file is a binary dictionary. Loading from a binary file is much
            faster than re-reading data files and saving is recommended

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

        save (bool):
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size

        return_data (bool):
            Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True


        Returns:
                dict: Dictionary with all modal heat fluxes requested
    """
    jm_path = __get_path(directory, inputfile)
    out_path = __get_path(directory, outputfile)
    data = __modal_analysis_read(nbins, nsamples, jm_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = __get_direction(directions)
    if 'x' in directions:
        out['jmxi'] = data[:, :, 0]
        out['jmxo'] = data[:, :, 1]
    if 'y' in directions:
        out['jmyi'] = data[:, :, 2]
        out['jmyo'] = data[:, :, 3]
    if 'z' in directions:
        out['jmz'] = data[:, :, 4]

    out['nbins'] = nbins
    out['nsamples'] = nsamples

    if save:
        np.save(out_path, out)

    if return_data:
        return out
    return


def load_kappamode(nbins, nsamples, directory=None,
                   inputfile='kappamode.out', directions='xyz',
                   outputfile='kappamode.npy', ndiv=None, save=False,
                   multiprocessing=False, ncore=None, block_size=65536, return_data=True):
    """
    Loads data from kappamode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory useage may be significantly larger than file size

    Args:
        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times thermal conductivity was sampled with HNEMA during GPUMD simulation

        directory (str):
            Name of directory storing the input file to read

        inputfile (str):
            Modal thermal conductivity file output by GPUMD

        directions (str):
            Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed (i.e. 'xz'
            is accepted)

        outputfile (str):
            File name to save read data to. Output file is a binary dictionary. Loading from a binary file is much
            faster than re-reading data files and saving is recommended

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

        save (bool):
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size

        return_data (bool):
            Toggle returning the loaded modal thermal conductivity data. If this is False, the user should ensure that
            save is True


        Returns:
                dict: Dictionary with all modal thermal conductivities requested
    """
    km_path = __get_path(directory, inputfile)
    out_path = __get_path(directory, outputfile)
    data = __modal_analysis_read(nbins, nsamples, km_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = __get_direction(directions)
    if 'x' in directions:
        out['kmxi'] = data[:, :, 0]
        out['kmxo'] = data[:, :, 1]
    if 'y' in directions:
        out['kmyi'] = data[:, :, 2]
        out['kmyo'] = data[:, :, 3]
    if 'z' in directions:
        out['kmz'] = data[:, :, 4]

    out['nbins'] = nbins
    out['nsamples'] = nsamples

    if save:
        np.save(out_path, out)

    if return_data:
        return out
    return


def load_saved_kappamode(filename='kappamode.npy', directory=None):
    """
    Loads data saved by the 'load_kappamode' function and returns the original dictionary.

    Args:
        filename (str):
            Name of the file to load

        directory (str):
            Directory the data file is located in

    Returns:
        dict: Dictionary with all modal thermal conductivities previously requested

    """
    path = __get_path(directory, filename)
    return np.load(path, allow_pickle=True).item()


def load_saved_heatmode(filename='heatmode.npy', directory=None):
    """
    Loads data saved by the 'load_heatmode' or 'get_gkma_kappa' function and returns the original dictionary.

    Args:
        filename (str):
            Name of the file to load

        directory (str):
            Directory the data file is located in

    Returns:
        dict: Dictionary with all modal heat flux previously requested

    """

    path = __get_path(directory, filename)
    return np.load(path, allow_pickle=True).item()


def load_sdc(Nc, directory=None, filename='sdc.out'):
    """
    Loads data from sdc.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Number of time correlation points the VAC/SDC is computed for

        directory (str):
            Directory to load 'sdc.out' file from (dir. of simulation)

        filename (str):
            File to load SDC from

    Returns:
        dict(dict):
            Dictonary with SDC/VAC data. The outermost dictionary stores each individual run

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t,VACx,VACy,VACz,SDCx,SDCy,SDCz
       **units**,ps,A^2/ps^2,A^2/ps^2,A^2/ps^2,A^2/ps,A^2/ps,A^2/ps

    """
    Nc = __check_list(Nc, varname='Nc', dtype=int)
    sdc_path = __get_path(directory, filename)
    data = pd.read_csv(sdc_path, delim_whitespace=True, header=None)
    __check_range(Nc, data.shape[0])
    labels = ['t', 'VACx', 'VACy', 'VACz', 'SDCx', 'SDCy', 'SDCz']

    start = 0
    out = dict()
    for i, npoints in enumerate(Nc):
        end = start + npoints
        run = dict()
        for j, key in enumerate(labels):
            run[key] = data[j][start:end].to_numpy()
        start = end
        out['run{}'.format(i)] = run
    return out


def load_vac(Nc, directory=None, filename='mvac.out'):
    """
    Loads data from mvac.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Number of time correlation points the VAC is computed for

        directory (str):
            Directory to load 'mvac.out' file from

        filename (str):
            File to load VAC from

    Returns:
        dict(dict):
            Dictonary with VAC data. The outermost dictionary stores each individual run

    Each run is a dictionary with keys:\n
    - t (ps)
    - VAC_x (Angstrom^2/ps^2)
    - VAC_y (Angstrom^2/ps^2)
    - VAC_z (Angstrom^2/ps^2)

    """
    Nc = __check_list(Nc, varname='Nc', dtype=int)
    vac_path = __get_path(directory, filename)
    with open(vac_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    start = 0
    for i, npoints in enumerate(Nc):
        run = dict()
        end = start + npoints
        if end > len(lines):
            raise IndexError("More data requested than exists.")

        run['t'] = np.zeros(npoints)
        run['VAC_x'] = np.zeros(npoints)
        run['VAC_y'] = np.zeros(npoints)
        run['VAC_z'] = np.zeros(npoints)
        for j, line in enumerate(lines[start:end]):
            data = line.split()
            run['t'][j] = float(data[0])
            run['VAC_x'][j] = float(data[1])
            run['VAC_y'][j] = float(data[2])
            run['VAC_z'][j] = float(data[3])
        start = end
        out['run{}'.format(i)] = run

    return out


def load_dos(num_dos_points, directory=None, filename='dos.out'):
    """
    Loads data from dos.out GPUMD output file.

    Args:
        num_dos_points (int or list(int)):
            Number of frequency points the DOS is computed for.

        directory (str):
            Directory to load 'dos.out' file from (dir. of simulation)

        filename (str):
            File to load DOS from.

    Returns:
        dict(dict)): Dictonary with DOS data. The outermost dictionary stores
        each individual run.

    Each run is a dictionary with keys:\n
    - nu (THz)
    - DOS_x (1/THz)
    - DOS_y (1/THz)
    - DOS_z (1/THz)

    """
    num_dos_points = __check_list(num_dos_points, varname='num_dos_points', dtype=int)
    dos_path = __get_path(directory, filename)
    with open(dos_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    start = 0
    for i, npoints in enumerate(num_dos_points):
        run = dict()
        end = start + npoints
        if end > len(lines):
            raise IndexError("More data requested than exists.")

        run['nu'] = np.zeros(npoints)
        run['DOS_x'] = np.zeros(npoints)
        run['DOS_y'] = np.zeros(npoints)
        run['DOS_z'] = np.zeros(npoints)
        for j, line in enumerate(lines[start:end]):
            data = line.split()
            run['nu'][j] = float(data[0])/(2*np.pi)
            run['DOS_x'][j] = float(data[1])
            run['DOS_y'][j] = float(data[2])
            run['DOS_z'][j] = float(data[3])
        start = end
        out['run{}'.format(i)] = run

    return out


def load_shc(Nc, num_omega, directory=None, filename='shc.out'):
    """
    Loads the data from shc.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Maximum number of correlation steps. If multiple shc runs, can provide a list of Nc.

        num_omega (int or list(int)):
            Number of frequency points. If multiple shc runs, can provide a list of num_omega.

        directory (str):
            Directory to load 'shc.out' file from (dir. of simulation)

        filename (str):
            File to load SHC from.

    Returns:
        dict: Dictionary of in- and out-of-plane shc results (average)


    Each run is a dictionary with keys:\n
    - t (ps)
    - K_in (ev*A/ps)
    - K_out (ev*A/ps)
    - nu (THz)
    - J_in (A*eV/ps/THz)
    - J_out (A*eV/ps/THz)
    """

    Nc = __check_list(Nc, varname='Nc',dtype=int)
    num_omega = __check_list(num_omega, varname='num_omega', dtype=int)
    if not len(Nc) == len(num_omega):
        raise ValueError('Nc and num_omega must be the same length.')
    shc_path = __get_path(directory, filename)

    with open(shc_path, 'r') as f:
        lines = f.readlines()

    out = dict()
    start = 0
    for i, varlen in enumerate(zip(Nc, num_omega)):
        Nc_i, num_omega_i = varlen
        ndata = 2*Nc_i-1
        end = start + ndata
        print(end, end + num_omega_i, len(lines))
        if end > len(lines) or end + num_omega_i > len(lines):
            raise IndexError("More data requested than exists.")
        run = dict()
        run['t'] = np.zeros(ndata)  # ps
        run['K_in'] = np.zeros(ndata)  # eV*A/ps
        run['K_out'] = np.zeros(ndata)  # eV*A/ps
        # correlation data
        for j, line in enumerate(lines[start:end]):
            data = line.split()
            run['t'][j] = float(data[0])
            run['K_in'][j] = float(data[1])
            run['K_out'][j] = float(data[2])
        start = end
        end += num_omega_i
        # spectral heat current
        run['nu'] = np.zeros(num_omega_i)  # THz
        run['J_in'] = np.zeros(num_omega_i)  # A*eV/ps/THz
        run['J_out'] = np.zeros(num_omega_i)  # A*eV/ps/THz
        for j, line in enumerate(lines[start:end]):
            data = line.split()
            run['nu'][j] = float(data[0])/(2*np.pi)
            run['J_in'][j] = float(data[1])
            run['J_out'][j] = float(data[2])
        start = end
        out['run{}'.format(i)] = run

    return out


def load_kappa(directory=None, filename='kappa.out'):
    """
    Loads data from kappa.out GPUMD output file which contains HNEMD kappa.

    Args:
        directory (str):
            Directory containing kappa data file

        filename (str):
            The kappa data file

    Returns:
        dict: A dictionary with keys corresponding to the columns in 'kappa.out'

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,kxi, kxo, kyi, kyo, kz
       **units**,W/m/K,W/m/K,W/m/K,W/m/K,W/m/K
    """

    kappa_path = __get_path(directory, filename)
    data = pd.read_csv(kappa_path, delim_whitespace=True, header=None)
    labels = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']
    out = dict()
    for i, key in enumerate(labels):
        out[key] = data[i].to_numpy()
    return out


def load_hac(Nc, output_interval, directory=None,filename='hac.out'):
    """
    Loads data from hac.out GPUMD output file.

    Args:
        Nc (int or list(int)):
            Number of correlation steps

        output_interval (int or list(int)):
            Output interval for HAC and RTC data

        directory (str):
            Directory containing hac data file

        filename (str):
            The hac data file

    Returns:
        dict: A dictionary containing the data from hac runs

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t, kxi, kxo, kyi, kyo, kz, jxijx, jxojx, jyijy, jyoJy, jzjz
       **units**,ps,W/m/K,W/m/K,W/m/K,W/m/K,W/m/K,ev^3/amu,ev^3/amu,ev^3/amu,ev^3/amu,ev^3/amu
    """

    Nc = __check_list(Nc, varname='Nc', dtype=int)
    output_interval = __check_list(output_interval, varname='output_interval', dtype=int)
    if not len(Nc) == len(output_interval):
        raise ValueError('Nc and output_interval must be the same length.')

    npoints = [int(x / y) for x, y in zip(Nc, output_interval)]
    hac_path = __get_path(directory, filename)
    data = pd.read_csv(hac_path, delim_whitespace=True, header=None)
    __check_range(npoints, data.shape[0])
    labels = ['t', 'jxijx', 'jxojx', 'jyijy', 'jyoJy', 'jzjz',
              'kxi', 'kxo', 'kyi', 'kyo', 'kz']
    start = 0
    out = dict()
    for i, varlen in enumerate(npoints):
        end = start + varlen
        run = dict()
        for j, key in enumerate(labels):
            run[key] = data[j][start:end].to_numpy()
        start = end
        out['run{}'.format(i)] = run
    return out


def get_frequency_info(bin_f_size, eigfile='eigenvector.out', directory=None):
    """
    Gathers eigen-frequency information from the eigenvector file and sorts
    it appropriately based on the selected frequency bins (identical to
    internal GPUMD representation).

    Args:
        bin_f_size (float):
            The frequency-based bin size (in THz)

        eigfile (str):
            The filename of the eigenvector output/input file created by GPUMD
            phonon package

        directory (str):
            Directory eigfile is stored

    Returns:
        dict: Dictionary with the system eigen-freqeuency information along
        with binning information

    """
    if not directory:
        eigpath = os.path.join(os.getcwd(), eigfile)
    else:
        eigpath = os.path.join(directory, eigfile)

    with open(eigpath, 'r') as f:
        om2 = [float(x) for x in f.readline().split()]

    fq = np.sign(om2) * np.sqrt(abs(np.array(om2))) / (2 * np.pi)
    fmax = (np.floor(np.abs(fq[-1]) / bin_f_size) + 1) * bin_f_size
    fmin = np.floor(np.abs(fq[0]) / bin_f_size) * bin_f_size
    shift = int(np.floor(np.abs(fmin) / bin_f_size))
    nbins = int(np.floor((fmax - fmin) / bin_f_size))
    bin_count = np.zeros(nbins)
    for freq in fq:
        bin_count[int(np.floor(np.abs(freq) / bin_f_size) - shift)] += 1
    return {'fq': fq, 'fmax': fmax, 'fmin': fmin, 'shift': shift,
            'nbins': nbins, 'bin_count': bin_count, 'bin_f_size': bin_f_size}


def reduce_frequency_info(freq, ndiv=1):
    """
    Recalculates frequency binning information based on how many times larger bins are wanted.

    Args:
        freq (dict): Dictionary with frequency binning information from the get_frequency_info function output

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

    Returns:
        dict: Dictionary with the system eigen freqeuency information along with binning information

    """
    freq = copy.deepcopy(freq)
    freq['bin_f_size'] = freq['bin_f_size'] * ndiv
    freq['fmax'] = (np.floor(np.abs(freq['fq'][-1]) / freq['bin_f_size']) + 1) * freq['bin_f_size']
    nbins_new = int(np.ceil(freq['nbins'] / ndiv))
    npad = nbins_new * ndiv - freq['nbins']
    freq['nbins'] = nbins_new
    freq['bin_count'] = np.pad(freq['bin_count'], [(0, npad)])
    freq['bin_count'] = np.sum(freq['bin_count'].reshape(-1, ndiv), axis=1)
    freq['ndiv'] = ndiv
    return freq

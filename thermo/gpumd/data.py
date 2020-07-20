import numpy as np
import pandas as pd
import os
import re
import copy
import multiprocessing as mp
from functools import partial
from collections import deque

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

#########################################
# Helper Functions
#########################################


def __get_direction(directions):
    """
    Creates a sorted list showing which directions the user asked for. Ex: 'xyz' -> ['x', 'y', 'z']

    Args:
        directions (str):
            A string containing the directions the user wants to process (Ex: 'xyz', 'zy', 'x')

    Returns:
        list(str): An ordered list that simplifies the user input for future processing

    """
    if not (bool(re.match('^[xyz]+$',directions))
            or len(directions) > 3
            or len(directions) == 0):
        raise ValueError('Invalid directions used.')
    return sorted(list(set(directions)))


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

def load_compute(directory=None, filename='compute.out', quantities=None):
    """
    loads data from compute.out GPUMD output file 

    Args:
        directory (str):
            Directory to load 'compute.out' file from (dir. of simulation)

        filename (str):
            file to load compute from

        quantities (str):
            allows user to set which quantities to extract from compute.out (such as temperature, force, jp, jk, etc.)

        Returns:
            'output' dictionary containing the data from compute.out
    """

    # test 1 was completed in 0.020443599999907747 seconds
    # test 2 was completed in 2.4768514999999987 seconds
    if not directory:
        com_n = os.path.join(os.getcwd(), filename)
    else:
        com_n = os.path.join(directory, filename)

    com_n = pd.read_csv(com_n, sep="\s+", header=None)

    total_cols = len(com_n.columns)
    q_count = {'temperature': 1, 'potential': 1, 'force': 3, 'virial': 3, 'jp': 3, 'jk': 3}
    output = dict()

    count = 0
    for value in quantities:
        count += q_count[value]
    if 'temperature' in quantities:
        m = int((total_cols - 2) / count)
    if 'temperature' not in quantities:
        m = int(total_cols / count)

    start = 0
    if 'temperature' in quantities:
        output['temperature'] = np.array(com_n.iloc[:, :m])
        output['heat_in'] = np.array(com_n.iloc[:, total_cols - 1:total_cols])
        output['heat_out'] = np.array(com_n.iloc[:, total_cols:])
        start = m

    if 'potential' in quantities:
        output['potential'] = np.array(com_n.iloc[:, start: m])
        start = m

    if 'force' in quantities:
        output['force'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start = start + (3 * m)

    if 'virial' in quantities:
        output['virial'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start = start + (3 * m)

    if 'jp' in quantities:
        output['jp'] = np.array(com_n.iloc[:, start: start + (3 * m)])
        start = start + (3 * m)

    if 'jk' in quantities:
        output['jk'] = np.array(com_n.iloc[:, start: start + (3 * m)])

    return output


def load_thermo(directory=None, filename='thermo.out',triclinic=False):
    """
    loads data from thermo.out GPUMD output file.

    Args:
        directory (str):
            Directory to load 'thermo.out' file from (dir. of simulation)

        filename (str):
            file to load thermo from

        triclinic (bool):
            allows user to set as true if triclinic, effects the total number of columns of data to add to.
            if triclinic is false, then orthogonal by default.

        Returns:
            'output' dictionary containing the data from thermo.out (ex: temperature, kinetic energy, etc.)
    """
    if not directory:
        t_path = os.path.join(os.getcwd(), filename)
    else:
        t_path = os.path.join(directory, filename)

    output = {'T': [], 'K': [], 'U': [], 'Px': [], 'Py': [], 'Pz': []}

    # orthogonal
    if not triclinic:
        output.update({'Lx': [], 'Ly': [], 'Lz': []})

    # triclinic
    else:
        output.update({'ax': [], 'ay': [], 'az': [], 'bx': [], 'by': [], 'bz': [], 'cx': [], 'cy': [], 'cz': []})

    with open(t_path) as f:
        for line in f:
            data = [float(num) for num in line.split()]
            for key in output.keys():
                index = list(output.keys()).index(key)
                output[key].append(data[index])
    return output


def load_heatmode(nbins, nsamples, directory=None,
                   inputfile='heatmode.out', directions='xyz',
                   outputfile='heatmode.npy', ndiv=None, save=False,
                   multiprocessing=False, ncore=None, block_size=65536, return_out=True):
    """
    Loads data from heatmode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory useage may be significantly larger than file size

    Args:
        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times heat flux was sampled with GKMA during GPUMD simulation

        directory (str):
            Name of directory storing the input file to read

        inputfile (str):
            Modal heat flux file output by GPUMD (default: heatmode.out)

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
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended (default:
            False)

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file (default: False)

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size (default: 2^16 = 65526)

        return_out (bool):
            Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True (default: True)


        Returns:
                dict: Dictionary with all modal heat fluxes requested
    """

    if not directory:
        jm_path = os.path.join(os.getcwd(), inputfile)
        out_path = os.path.join(os.getcwd(), outputfile)
    else:
        jm_path = os.path.join(directory, inputfile)
        out_path = os.path.join(directory, outputfile)

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

    if return_out:
        return out
    return


def load_kappamode(nbins, nsamples, directory=None,
                   inputfile='kappamode.out', directions='xyz',
                   outputfile='kappamode.npy', ndiv=None, save=False,
                   multiprocessing=False, ncore=None, block_size=65536, return_out=True):
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
            Modal thermal conductivity file output by GPUMD (default: kappamode.out)

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
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended (default:
            False)

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file (default: False)

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size (default: 2^16 = 65526)

        return_out (bool):
            Toggle returning the loaded modal thermal conductivity data. If this is False, the user should ensure that
            save is True (default: True)


        Returns:
                dict: Dictionary with all modal thermal conductivities requested
    """

    if not directory:
        km_path = os.path.join(os.getcwd(), inputfile)
        out_path = os.path.join(os.getcwd(), outputfile)
    else:
        km_path = os.path.join(directory, inputfile)
        out_path = os.path.join(directory, outputfile)

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

    if return_out:
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

    if directory:
        path = os.path.join(directory, filename)
    else:
        path = os.path.join(os.getcwd(), filename)
    return np.load(path, allow_pickle=True).item()


def load_saved_heatmode(filename='heatmode.npy', directory=None):
    """
    Loads data saved by the 'load_heatmode' function and returns the original dictionary.

    Args:
        filename (str):
            Name of the file to load

        directory (str):
            Directory the data file is located in

    Returns:
        dict: Dictionary with all modal heat flux previously requested

    """

    if directory:
        path = os.path.join(directory, filename)
    else:
        path = os.path.join(os.getcwd(), filename)
    return np.load(path, allow_pickle=True).item()


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
    if not is_int and average:
        raise ValueError('average cannot be used if Nc is not an int.')

    if not is_int and len(Nc) != num_run:
        raise ValueError('length of Nc must be equal to num_run.')

    if not is_int and len(Nc) == 1:
        Nc = Nc[0]

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
        vrun['t'] = np.zeros(pt_rng)
        srun['t'] = np.zeros(pt_rng)

        vrun['VAC_x'] = np.zeros(pt_rng)
        vrun['VAC_y'] = np.zeros(pt_rng)
        vrun['VAC_z'] = np.zeros(pt_rng)

        srun['SDC_x'] = np.zeros(pt_rng)
        srun['SDC_y'] = np.zeros(pt_rng)
        srun['SDC_z'] = np.zeros(pt_rng)
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
        pt_rng = Nc # Required for average, checked above
        vave = dict()
        save = dict()

        vave['t'] = np.zeros(pt_rng)
        vave['VAC_x'] = np.zeros(pt_rng)
        vave['VAC_y'] = np.zeros(pt_rng)
        vave['VAC_z'] = np.zeros(pt_rng)

        save['t'] = np.zeros(pt_rng)
        save['SDC_x'] = np.zeros(pt_rng)
        save['SDC_y'] = np.zeros(pt_rng)
        save['SDC_z'] = np.zeros(pt_rng)

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
    if not is_int and average:
        raise ValueError('average cannot be used if Nc is not an int.')

    if not is_int and len(Nc) != num_run:
        raise ValueError('length of Nc must be equal to num_run.')

    if not is_int and len(Nc) == 1:
        Nc = Nc[0]

    if directory == '':
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
        run['t'] = np.zeros(pt_rng)
        run['VAC_x'] = np.zeros(pt_rng)
        run['VAC_y'] = np.zeros(pt_rng)
        run['VAC_z'] = np.zeros(pt_rng)
        for point in range(pt_rng):
            data = lines[idx_shift + point].split()
            run['t'][point] = float(data[0])
            run['VAC_x'][point] = float(data[1])
            run['VAC_y'][point] = float(data[2])
            run['VAC_z'][point] = float(data[3])
        idx_shift += pt_rng

        out['run'+str(run_num)] = run

    if average:
        pt_rng = Nc  # Required for average, checked above
        ave = dict()
        ave['t'] = np.zeros(pt_rng)
        ave['VAC_x'] = np.zeros(pt_rng)
        ave['VAC_y'] = np.zeros(pt_rng)
        ave['VAC_z'] = np.zeros(pt_rng)

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
    if not is_int and average:
        raise ValueError('average cannot be used if points_per_run is not an int.')

    if not is_int and len(points_per_run) != num_run:
        raise ValueError('length of points_per_run must be equal to num_run.')

    if not is_int and len(points_per_run) == 1:
        points_per_run = points_per_run[0]

    if directory == '':
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
        run['nu'] = np.zeros(pt_rng)
        run['DOS_x'] = np.zeros(pt_rng)
        run['DOS_y'] = np.zeros(pt_rng)
        run['DOS_z'] = np.zeros(pt_rng)
        for point in range(pt_rng):
            data = lines[idx_shift + point].split()
            run['nu'][point] = float(data[0])/(6.283185307179586)
            run['DOS_x'][point] = float(data[1])
            run['DOS_y'][point] = float(data[2])
            run['DOS_z'][point] = float(data[3])
        idx_shift += pt_rng

        out['run'+str(run_num)] = run

    if average:
        pt_rng = points_per_run  # required for average, checked above
        ave = dict()
        ave['nu'] = np.zeros(pt_rng)
        ave['DOS_x'] = np.zeros(pt_rng)
        ave['DOS_y'] = np.zeros(pt_rng)
        ave['DOS_z'] = np.zeros(pt_rng)

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
    if not type(Nc) == int:
        raise ValueError('Nc must be an int.')
        
    if directory == '':
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
    shc_in = np.mean(shc_in,0)*1000./10.18 # eV*A/ps
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
        t = np.zeros((N, 1))
        x_ac_i = np.zeros((N, 1))  # autocorrelation IN, X
        x_ac_o = np.zeros((N, 1))  # autocorrelation OUT, X

        y_ac_i = np.zeros((N, 1))  # autocorrelation IN, Y
        y_ac_o = np.zeros((N, 1)) # autocorrelation OUT, Y

        z_ac = np.zeros((N, 1))  # autocorrelation Z

        kx_i = np.zeros((N, 1))  # kappa IN, X
        kx_o = np.zeros((N, 1))  # kappa OUT, X

        ky_i = np.zeros((N, 1))  # kappa IN, Y
        ky_o = np.zeros((N, 1))  # kappa OUT, Y

        kz = np.zeros((N, 1))  # kappa, Z

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

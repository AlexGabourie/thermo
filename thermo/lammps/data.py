import os
import numpy as np

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def __get_path(directory, filename):
    if not directory:
        return os.path.join(os.getcwd(), filename)
    return os.path.join(directory, filename)


def __process_box(out, filehandle):
    """
    Processes the lines of a trajectory file dedicated to the box size and shape

    Args:
        out (dict):
            Stores all relevant box information

        filehandle (TextIOWrapper):
            The trajectory file handle

    """
    for pair in [('x', 'xy'), ('y', 'xz'), ('z', 'yz')]:
        data = [float(i) for i in filehandle.readline().split()]
        out[pair[0]].append(data[1]-data[0])
        triclinic = len(data) == 3
        out[pair[1]].append(data[2] if triclinic else None)

    if triclinic:
        a = np.array([out['x'][-1], 0, 0])
        b = np.array([out['xy'][-1], out['y'], 0])
        c = np.array([out['xz'][-1], out['yz'], out['z']])
        Avec = np.cross(a, b)
        out['A'].append(np.linalg.norm(Avec))
        out['V'].append(np.abs(np.dot(Avec, c)))
    else:
        out['A'].append(out['x'][-1]*out['y'][-1])
        out['V'].append(out['A'][-1]*out['z'][-1])


def get_dimensions(filename, directory=None):
    """
    Gets the dimensions of a 3D simulation from a LAMMPS trajectory.

    Args:
        filename (str):
            LAMMPS trajectory file to extract dimensions from

        directory (str):
            The directory the trajectory file is found in

    Returns:
        dict: Dictionary with keys given in the table below

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**, x, y, z, A, V, xy, xz, yz
       **units**,|d1|,|d1|,|d1|,|d2|,|d3|,|d1|,|d1|,|d1|

    .. |d1| replace:: distance
    .. |d2| replace:: distance\ :sup:`2`
    .. |d3| replace:: distance\ :sup:`3`

    """
    trjpath = __get_path(directory, filename)
    labels = ['x', 'y', 'z', 'A', 'V', 'xy', 'xz', 'yz']
    out = dict()
    for label in labels:
        out[label] = []

    with open(trjpath) as f:
        line = f.readline()
        while line:
            if 'BOX' in line:
                __process_box(out, f)
            line = f.readline()
    return out


def extract_dt(log_file):
    """
    Finds all time steps given in the lammps output log

    Args:
        log_file (str):
            LAMMPS log file to examine

    Returns:
        list(float): The timesteps found in log_file in units of time
    """
    dt = list()
    if os.path.isfile(log_file):
        with open(log_file, 'r') as log:
            lines = log.readlines()

        for line in lines:
            elements = line.split()
            if len(elements) > 0 and ' '.join(elements[0:2]) == 'Time step':
                dt.append(float(elements[3]))
        if len(dt) == 0:
            print('No timesteps found in', log_file)
    else:
        print(log_file, 'not found')

    return dt

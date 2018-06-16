
from pylab import *
import numpy as np


def load_forces(force_file, type):
    '''

    :param force_file: If type == 'LAMMPS':
        The file path for the LAMMPS output forces
         LAMMPS file should be in the format given by the following LAMMPS input command:
         force all custom 1 <file> id fx fy fz
        If type == 'GPUMD':
        the force output file (f.out) path when GPUMD is compiled with the force flag
    :return: dictionary containing sorted force vectors
    '''

    # Load force outputs
    if type == 'LAMMPS':
        # LAMMPS
        with open(force_file, 'r') as f:
            llines = f.readlines()

        # remove header info
        llines = llines[9:]

        # process atomic forces
        n = len(llines)
        xf, yf, zf = np.zeros(n), np.zeros(n), np.zeros(n)
        for line in llines:
            num = line.split()
            ID = int(num[0])-1
            xf[ID] = float(num[1])
            yf[ID] = float(num[2])
            zf[ID] = float(num[3])

    elif type == 'GPUMD':
        # GPUMD
        with open(force_file, 'r') as f:
            glines = f.readlines()

        # process atomic forces
        xf, yf, zf = list(), list(), list()
        for line in glines:
            num = line.split()
            xf.append(float(num[0]))
            yf.append(float(num[1]))
            zf.append(float(num[2]))

        xf = np.array(xf)
        yf = np.array(yf)
        zf = np.array(zf)
    else:
        raise ValueError('Invalid simulation type passed. Forces not extracted.')

    # fill return dictionary
    out = dict()
    out['xf'] = xf
    out['yf'] = yf
    out['zf'] = zf
    return out

def compare_forces(f1_dict, f2_dict):
    '''

    :param f1_dict: dictionary containing extracted forces from a GPUMD or LAMMPS simulation
    :param f2_dict: dictionary containing extracted forces from a GPUMD or LAMMPS simulation
    :return: comparison dictionary

    Forces are dict2 - dict1 values
    '''

    out = dict()
    out['xdiff'] = f2_dict['xf'] - f1_dict['xf']
    out['ydiff'] = f2_dict['yf'] - f1_dict['yf']
    out['zdiff'] = f2_dict['zf'] - f1_dict['zf']
    out['xnorm'] = np.linalg.norm(out['xdiff'])
    out['ynorm'] = np.linalg.norm(out['ydiff'])
    out['znorm'] = np.linalg.norm(out['zdiff'])
    return out


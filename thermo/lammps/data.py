import os
import re
import sys

def get_sim_dimensions(lammpstrj_file):
    '''
    Reads a LAMMPS trajectory file and extracts simulation dimensions.

    Args:
        lammpstrj_file (str):
            LAMMPS trajectory file to extract dimensions from

    Returns:
        out (dict):
        - x (list(float)): x-dimension [angstroms]
        - y (list(float)): y-dimension [angstroms]
        - z (list(float)): z-dimension [angstroms]
        - area (list(float)): [angstroms^2]
        - volume (list(float)): [anstroms^3]

    '''

    if os.path.isfile(lammpstrj_file):
        area = list()
        x = list()
        y = list()
        z = list()
        volume = list()
        flag = 0
        cnt = 0

        xdata = -1
        ydata = -1
        zdata = -1

        with open(lammpstrj_file, 'r') as trj:
            for line in trj:
                if flag == 1:
                    if cnt == 0:
                        xdata = re.findall('[-+]?\d+\.\d+e.\d+', line)
                        if not xdata:
                            xdata = re.findall('[-+]?\d+\.\d+', line)
                        cnt += 1
                    elif cnt == 1:
                        ydata = re.findall('[-+]?\d+\.\d+e.\d+', line)
                        if not ydata:
                            ydata = re.findall('[-+]?\d+\.\d+', line)
                            area.append(xwidth*ywidth)
                        cnt += 1
                    elif cnt == 2:
                        zdata = re.findall('[-+]?\d+\.\d+e.\d+', line)
                        if not zdata:
                            zdata = re.findall('[-+]?\d+\.\d+', line)
                        if not (len(ydata) == 1 or len(xdata) == 1 or len(zdata) == 1):
                            xwidth = float(xdata[1])-float(xdata[0])
                            ywidth = float(ydata[1])-float(ydata[0])
                            zwidth = float(zdata[1])-float(zdata[0])
                            area.append(xwidth*ywidth)
                            volume.append(xwidth*ywidth*zwidth)
                            x.append(xwidth)
                            y.append(ywidth)
                            z.append(zwidth)
                        cnt = 0
                        flag = 0
                if not len(re.findall('BOX', line)) == 0:
                    flag = 1

        return {'x':x, 'y':y, 'z':z, 'area':area, 'volume':volume}
    else:
        raise Error('file {} not found'.format(lammpstrj_file))


def extract_dt(log_file):
    '''
    Finds all time steps given in the lammps output log

    Args:
        log_file (str):
            LAMMPS log file to examine

    Returns:
        dt (list(flaat)):
            The timesteps found in log_file in [ps]
    '''
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

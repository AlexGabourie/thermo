import os
import scipy.io as sio
import numpy as np
import sys

def get_heat_flux(**kwargs):
    
    original_dir = os.getcwd()

    try:
        # Get arguments
        directory = kwargs['directory'] if 'directory' in kwargs.keys() else './'
        heatflux_file = kwargs['heatflux_file'] if 'heatflux_file' in kwargs.keys() else os.path.join(directory, 'heat_out.heatflux')
        mat_file = kwargs['mat_file'] if 'mat_file' in kwargs.keys() else os.path.join(directory, 'heat_flux.mat')
        
        # Check that directory exists
        if not os.path.isdir(directory):
            raise IOError('The path: {} is not a directory.'.format(directory))
            
        # Go to directory and see if imported .mat file already exists
        os.chdir(directory)
        if os.path.isfile(mat_file) and mat_file.endswith('.mat'):
            return sio.loadmat(mat_file)
        
        # Continue with the import since .mat file 
        if not os.path.isfile(heatflux_file):
            raise IOError('The file: \'{}{}\' is not found.'.format(directory,heatflux_file))
        
        # Read the file
        with open(heatflux_file, 'r') as hf_file:
            lines = hf_file.readlines()[2:]
         
        # Get timestep
        rate = int(lines[0].split()[0])

        # read all data
        jx = list()
        jy = list()
        jz = list()
        for line in lines:
            vals = line.split()
            jx.append(float(vals[1]))
            jy.append(float(vals[2]))
            jz.append(float(vals[3]))

        output = {'Jx':jx, 'Jy':jy, 'Jz':jz, 'rate':rate}
        sio.savemat(mat_file, output)
        os.chdir(original_dir)
        return output

    except:
        os.chdir(original_dir)
        print(sys.exc_info()[0])


    
        
    

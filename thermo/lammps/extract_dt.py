import os

def extract_dt(log):
    '''Finds all time steps given in the lammps output log'''
    dt = list()
    if os.path.isfile(log):
        with open(log, 'r') as log_file:
            lines = log_file.readlines()
    
        for line in lines:
            elements = line.split()
            if len(elements) > 0 and ' '.join(elements[0:2]) == 'Time step':
                dt.append(float(elements[3]))


        if len(dt) == 0:
            print('No timesteps found in', log)

    else:
        print(log, 'not found')
        
    return dt
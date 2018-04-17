
def get_gpumd_tc(directory=''):
    import matplotlib as plt
    import numpy as np
    import os

    if directory=='':
        hac_path = os.path.join(os.getcwd(),'hac.out')
    else:
        hac_path = os.path.join(directory,'hac.out')

    with open(hac_path, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        t = np.zeros((N,1))
        x_ac_i = np.zeros((N,1)) # autocorrelation IN, X
        x_ac_o = np.zeros((N,1)) # autocorrelation OUT, X
        x_ac_c = np.zeros((N,1)) # cross-correlation IN/OUT, X

        y_ac_i = np.zeros((N,1)) # autocorrelation IN, Y
        y_ac_o = np.zeros((N,1)) # autocorrelation OUT, Y
        y_ac_c = np.zeros((N,1)) # cross-correlation IN/OUT, Y

        z_ac = np.zeros((N,1)) # autocorrelation Z

        kx_i = np.zeros((N,1)) # kappa IN, X
        kx_o = np.zeros((N,1)) # kappa OUT, X
        kx_c = np.zeros((N,1)) # kappa cross, X

        ky_i = np.zeros((N,1)) # kappa IN, Y
        ky_o = np.zeros((N,1)) # kappa OUT, Y
        ky_c = np.zeros((N,1)) # kappa cross, Y

        kz = np.zeros((N,1)) # kappa, Z

        for i, line in enumerate(lines):
            vals = line.split()
            t[i] = vals[0]
            x_ac_i[i] = vals[1]
            x_ac_o[i] = vals[2]
            x_ac_c[i] = vals[3]
            y_ac_i[i] = vals[4]
            y_ac_o[i] = vals[5]
            y_ac_c[i] = vals[6]
            z_ac[i] = vals[7]
            kx_i[i] = vals[8]
            kx_o[i] = vals[9]
            kx_c[i] = vals[10]
            ky_i[i] = vals[11]
            ky_o[i] = vals[12]
            ky_c[i] = vals[13]
            kz[i] = vals[14]

    out = dict()
    out['hac_i'] = (x_ac_i + y_ac_i)/2.
    out['hac_o'] = (x_ac_o + y_ac_o)/2.
    out['tc_i'] = (kx_i + ky_i)/2.
    out['tc_o'] = (kx_o + ky_o)/2.
    out['tc_c'] = (kx_c + ky_c)/2.
    out['tc_z'] = kz
    out['tc_t'] = tc_i+tc_o+tc_c+tc_z
    out['t'] = t

    return out

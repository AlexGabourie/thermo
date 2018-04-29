def plot_gpumd_tc():
    get_ipython().magic(u'matplotlib notebook')
    import matplotlib as plt
    import numpy as np

    with open('hac.out', 'r') as f:
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


    hac_i = (x_ac_i + y_ac_i)/2.
    hac_o = (x_ac_o + y_ac_o)/2.
    tc_i = (kx_i + ky_i)/2.
    tc_o = (kx_o + ky_o)/2.
    tc_c = (kx_c + ky_c)/2.
    tc_z = kz



    plt.figure()
    plt.loglog(t,hac_i/hac_i[0])
    plt.loglog(t,hac_o/hac_o[0])
    plt.xlabel('Correlation Time (ps)')
    plt.ylabel('Normalized HAC')
    plt.legend(['In', 'Out'])
    plt.ylim([10**(-4), 10])

    plt.figure()
    plt.plot(t, tc_i, linestyle='-')
    plt.plot(t, tc_o, linestyle='--')
    plt.plot(t, tc_c, linestyle='-.')
    plt.plot(t, tc_z, linestyle=':')
    plt.plot(t, tc_i+tc_o+tc_c+tc_z)
    plt.xlabel('Correlation Time (ps)')
    plt.ylabel('Thermal Conductivity (W/m/K)')
    plt.legend(['In', 'Out', 'Cross', 'z', 'Total'])
    plt.xlim([0, max(t)])
    plt.ylim([-20, 130])
    plt.show()


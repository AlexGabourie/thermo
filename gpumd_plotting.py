from pylab import *

def plot_gpumd_tc():
    with open('hac.out', 'r') as f:
        lines = f.readlines()
        N = len(lines)
        t = zeros((N,1))
        x_ac_i = zeros((N,1)) # autocorrelation IN, X
        x_ac_o = zeros((N,1)) # autocorrelation OUT, X
        x_ac_c = zeros((N,1)) # cross-correlation IN/OUT, X

        y_ac_i = zeros((N,1)) # autocorrelation IN, Y
        y_ac_o = zeros((N,1)) # autocorrelation OUT, Y
        y_ac_c = zeros((N,1)) # cross-correlation IN/OUT, Y

        z_ac = zeros((N,1)) # autocorrelation Z

        kx_i = zeros((N,1)) # kappa IN, X
        kx_o = zeros((N,1)) # kappa OUT, X
        kx_c = zeros((N,1)) # kappa cross, X

        ky_i = zeros((N,1)) # kappa IN, Y
        ky_o = zeros((N,1)) # kappa OUT, Y
        ky_c = zeros((N,1)) # kappa cross, Y

        kz = zeros((N,1)) # kappa, Z

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



    figure()
    loglog(t,hac_i/hac_i[0])
    loglog(t,hac_o/hac_o[0])
    xlabel('Correlation Time (ps)')
    ylabel('Normalized HAC')
    legend(['In', 'Out'])
    ylim([10**(-4), 10])

    figure()
    plot(t, tc_i, linestyle='--')
    plot(t, tc_o, linestyle='--')
    plot(t, tc_c, linestyle='-.')
    plot(t, tc_z, linestyle=':')
    plot(t, tc_i+tc_o+tc_c+tc_z)
    xlabel('Correlation Time (ps)')
    ylabel('Thermal Conductivity (W/m/K)')
    legend(['In', 'Out', 'Cross', 'z', 'Total'])
    xlim([0, max(t)])
    ylim([-20, 130])
    show()


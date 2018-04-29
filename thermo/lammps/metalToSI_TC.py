def metalToSI_TC( vol, T ):
    '''Converts LAMMPS metal units to SI units for thermal conductivity calculations.'''
    kb = 1.38064852e-23 #m^3*kg/(s^2*K)
    vol = vol/(1.0e10)**3 #to m^3
    #eV^2*ns/(ps^2*angstrom^4) to J^2/(s*m^4)
    to_SI = (1.602e-19)**2.*1.0e12*(1.0e10)**4.0*1000. 
    return vol*to_SI/(kb*T**2)
from ase.io import read
import atomman
from ase import Atom, Atoms


def atoms2lammps(atoms, out_file='atoms.data', add_masses = True):
    """
    Converts ASE atoms to a lammps data file

    Parameters
    ----------
    arg1 : atoms
        Atoms to write to lammps data file

    arg2 : gpumd_file
        file to save the structure data to

    Returns
    -------

    """
    sys, elem = atomman.load.ase_Atoms.load(atoms)
    # write data file
    atomman.lammps.atom_data.dump(sys, out_file)

    if add_masses:
        # Write block of string for mass inclusion
        mass_str = 'Masses\n\n'
        for i, element in enumerate(elem):
            mass_str += '{} {}\n'.format(str(i + 1), str(Atom(element).mass))

        mass_str += '\n'

        # add the mass string to the correct part of the datafile
        with open(out_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'Atoms' in line:
                break

        lines.insert(i, mass_str)

        with open(out_file, 'w') as f:
            f.write(''.join(lines))



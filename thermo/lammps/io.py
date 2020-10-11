import atomman
from ase import Atom

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def ase_atoms_to_lammps(atoms, out_file='atoms.data', add_masses=True):
    """
    Converts ASE atoms to a lammps data file.

    Args:
        atoms (ase.Atoms):
            Atoms to write to lammps data file

        out_file (str):
            File to save the structure data to

        add_masses (Bool):
            Determines if atom masses are written to data file

    """
    sys = atomman.load_ase_Atoms(atoms)
    elem = sys.symbols
    # write data file
    atomman.dump_atom_data(sys, out_file)

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

        lines.insert(len(lines)-1, mass_str)

        with open(out_file, 'w') as f:
            f.write(''.join(lines))

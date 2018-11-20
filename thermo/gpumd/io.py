from ase.io import write
from ase.io import read
from ase import Atom, Atoms
from math import floor

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


#########################################
# Read Related
#########################################

def load_xyz(filename='xyz.in', atom_types=None):
    """
    Reads and returns the structure input file from GPUMD.

    Args:
        filename (str):
            Name of structure file

        atom_types (list(str)):
            List of atom types (elements).

    Returns:
        atoms (ase.Atoms):
            ASE atoms object with x,y,z, mass, group, type, cell, and PBCs
            from input file. group is stored in tag, atom type may not
            correspond to correct atomic symbol
        M (int):
            Max number of neighbor atoms

        cutoff (float):
            Initial cutoff for neighbor list build
    """
    # read file
    with open(filename) as f:
        xyz_lines = f.readlines()

    # get global structure params
    N, M, cutoff = tuple(xyz_lines[0].split())
    N, M, cutoff = int(N), int(M), float(cutoff)
    pbc_x, pbc_y, pbc_z, L_x, L_y, L_z = tuple(xyz_lines[1].split())
    pbc_x, pbc_y, pbc_z, L_x, L_y, L_z = int(pbc_x), int(pbc_y), int(pbc_z), \
                                            float(L_x), float(L_y), float(L_z)

    # get atomic params
    atoms = Atoms()
    atoms.set_pbc((pbc_x, pbc_y, pbc_z))
    atoms.set_cell([(L_x, 0, 0), (0, L_y, 0), (0, 0, L_z)])
    for line in xyz_lines[2:]:
        type_, group, mass, x, y, z = tuple(line.split())
        atoms.append(Atom(int(type_), (float(x), float(y), float(z)),
                          tag=int(group), mass=float(mass)))

    if atom_types:
        __set_atoms(atoms, atom_types)

    return atoms, M, cutoff

def __set_atoms(atoms, types):
    """
    Sets the atom symbols for atoms loaded from GPUMD where in.xyz does not
    contain that information

    Args:
        atoms (ase.Atoms):
            Atoms object to change symbols in

        types (list(str)):
            List of strings to assign to atomic symbols

    """
    for atom in atoms:
        atom.symbol = types[atom.number]

def load_traj(traj_file='xyz.out', in_file='xyz.in'):
    """
    Reads the trajectory from GPUMD run and creates a list of ASE atoms.

    Args:
        traj_file (str):
            Name of the file that hold the GPUMD trajectory

        in_file (str):
            Name of the original structure input file. Needed to get atom
            type, mass, etc

    Returns:
        traj (list(ase.Atoms)):
            A list of ASE atoms objects.
    """
    # read trajectory file
    with open(traj_file, 'r') as f:
        xyz_lines = f.readlines()

    atoms_in, M, cutoff = load_xyz(in_file)
    N = len(atoms_in)

    num_frames = len(xyz_lines)/float(N)
    if not (num_frames == floor(num_frames)):
        print('load_traj warning: Non-integer number of frames base on number of atoms.' +
              ' Only taking {} frames'.format(floor(num_frames)))

    num_frames = int(floor(num_frames))
    traj = list()
    for frame in range(num_frames):
        for i, line in enumerate(xyz_lines[frame*N:(frame+1)*N]):
            curr_atom = atoms_in[i]
            curr_atom.position = tuple([float(val) for val in line.split()])
        traj.append(atoms_in.copy())

    return traj

#########################################
# Write Related
#########################################

def convert_gpumd_atoms(in_file='xyz.in', out_filename='in.xyz',
                            format='xyz', atom_types=None):
    """
    Converts the GPUMD input structure file to any compatible ASE
    output structure file

    Args:
        in_file (str):
            GPUMD position file to get structure from

        out_filename (str):
            Name of output file after conversion

        format (str):
            ASE supported output format

        atom_types (list(str)):
            List of atom types (elements).

    """
    atoms, M, cutoff = load_xyz(in_file, atom_types)
    write(out_filename, atoms, format)
    return

def convert_gpumd_traj(traj_file='xyz.out', out_filename='out.xyz',
                       in_file='xyz.in', format='xyz'):
    """
    Converts GPUMD trajectory to any compatible ASE output. Default: xyz

    Args:
        traj_file (str):
            Trajetory from GPUMD

        out_filename (str):
            File in which final trajectory should be saved

        in_file (str):
            Original stucture input file to GPUMD. Needed to get atom
            numbers/types

        format (str):
            ASE supported format

    """
    traj = load_traj(traj_file, in_file)
    write(out_filename, traj, format)
    return

def lammps_atoms_to_gpumd(filename, M, cutoff, style='atomic',
                        gpumd_file='xyz.in'):
    """
    Converts a lammps data file to GPUMD compatible position file

    Args:
        filename (str):
            LAMMPS data file name

        M (int):
            Maximum number of neighbors for one atom

        cutoff (float):
            Initial cutoff distance for building the neighbor list

        style (str):
            Atom style used in LAMMPS data file

        gpumd_file (str):
            File to save the structure data to

    """
    # Load atoms
    atoms = read(filename, format='lammps-data', style=style)
    ase_atoms_to_gpumd(atoms, M, cutoff, gpumd_file=gpumd_file)
    return


def __atoms_sortkey(atom, atom_order=None):
    """
    Used as a key for sorting atoms into groups or types for GPUMD in.xyz files

    Args:
        atom (ase.Atom):
            Atom object

        atom_order (list(str)):
            A list of atomic symbol strings in the desired order.
            If None, atom tag is used for sorting (NEMD)

    """
    if atom_order:
        for i, sym in enumerate(atom_order):
            if sym == atom.symbol:
                return i
    else:
        return atom.tag

def ase_atoms_to_gpumd(atoms, M, cutoff, gpumd_file='xyz.in', sort_key=None,
        atom_order=None):
    """
    Converts ASE atoms to GPUMD compatible position file

    Args:
        atoms (ase.Atoms):
            Atoms to write to gpumd file

        M (int):
            Maximum number of neighbors for one atom

        cutoff (float):
            Initial cutoff distance for building the neighbor list

        gpumd_file (str):
            File to save the structure data to

        sort_key (str):
            How to sort atoms ('group', 'type'). Default is None.

        atom_order (list(str)):
            List of atomic symbols in order to be listed in GPUMD xyz file.
            Default is None

    """

    if sort_key == 'type':
        atoms_list = sorted(atoms, key=lambda x: __atoms_sortkey(x, atom_order))
    elif sort_key == 'group':
        atoms_list = sorted(atoms, key=lambda x: __atoms_sortkey(x))
    else:
        atoms_list = atoms

    if sort_key=='type' and atom_order:
        types = atom_order
    else:
        types = list(set(atoms.get_chemical_symbols()))

    type_dict = dict()
    for i, type_ in enumerate(types):
        type_dict[type_] = i

    N = len(atoms)
    pbc = [str(1) if val else str(0) for val in atoms.get_pbc()]
    lx, ly, lz, a1, a2, a3 = tuple(atoms.get_cell_lengths_and_angles())
    lx, ly, lz = str(lx), str(ly), str(lz)
    if not (a1 == a2 == a3):
        raise ValueError('Structure must be orthorhombic.')

    with open(gpumd_file, 'w') as f:
        f.writelines(' '.join([str(N), str(M), str(cutoff)]) + '\n')
        f.writelines(' '.join(pbc + [lx, ly, lz]) + '\n')
        for atom in atoms_list[:-1]:
            type_ = [type_dict[atom.symbol], atom.tag, atom.mass] + list(atom.position)
            f.writelines(' '.join([str(val) for val in type_]) + '\n')
        # Last line
        atom = atoms[-1]
        type_ = [type_dict[atom.symbol], atom.tag, atom.mass] + list(atom.position)
        f.writelines(' '.join([str(val) for val in type_]))
    return

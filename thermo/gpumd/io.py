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

    Parameters
    ----------
    arg1 : filename
        name of structure file

    Returns
    -------
    atoms
        ASE atoms object with x,y,z, mass, group, type, cell, and PBCs
        from input file. group is stored in tag, atom type may not
        correspond to correct atomic symbol
    M
        Max number of neighbor atoms

    cutoff
        initial cutoff for neighbor list build
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

    Parameters
    ----------
    arg1 : atoms
        Atoms object to change symbols in

    arg2 : types
        list of strings to assign to atomic symbols

    """
    for atom in atoms:
        atom.symbol = types[atom.number]

def load_traj(traj_file='xyz.out', in_file='xyz.in'):
    """
    Reads the trajectory from GPUMD run and creates a list of ASE atoms.

    Parameters
    ----------
    arg1 : traj_file
        name of the file that hold the GPUMD trajectory

    arg2 : in_file
        name of the original structure input file. Needed to get atom type, mass, etc


    Returns
    -------
    traj
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
                            format='xyz'):
    """
    Converts the GPUMD input structure file to any compatible ASE
    output structure file

    Parameters
    ----------
    arg1 : in_file
        GPUMD position file to get structure from

    arg2 : out_filename
        name of output file after conversion

    arg3 : format
        ASE supported output format

    Returns
    -------

    """
    atoms, M, cutoff = load_xyz(in_file)
    write(out_filename, atoms, format)
    return

def convert_gpumd_traj(traj_file='xyz.out', out_filename='out.xyz',
                       in_file='xyz.in', format='xyz'):
    """
    Converts GPUMD trajectory to any compatible ASE output. Default: xyz

    Parameters
    ----------
    arg1 : traj_file
        trajetory from GPUMD

    arg2 : out_filename
        file in which final trajectory should be saved

    arg3 : in_file
        original stucture input file to GPUMD. Needed to get atom numbers/types

    arg4 : format
        ASE supported format

    Returns
    -------

    """
    traj = load_traj(traj_file, in_file)
    write(out_filename, traj, format)
    return

def lammps_atoms_to_gpumd(filename, M, cutoff, style='atomic',
                        gpumd_file='xyz.in'):
    """
    Converts a lammps data file to GPUMD compatible position file

    Parameters
    ----------
    arg1 : filename
        LAMMPS data file name

    arg2 : M
        Maximum number of neighbors for one atom

    arg3 : cutoff
        initial cutoff distance for building the neighbor list

    arg4 : style
        atom style used in LAMMPS data file

    arg5 : gpumd_file
        file to save the structure data to

    Returns
    -------

    """
    # Load atoms
    atoms = read(filename, format='lammps-data', style=style)
    ase_atoms_to_gpumd(atoms, M, cutoff, gpumd_file=gpumd_file)
    return


def __atoms_sortkey(atom, atom_order=None):
    """
    Used as a key for sorting atoms into groups or types for GPUMD in.xyz files

    Parameters
    ----------
    arg1 : atom
        Atom object

    arg2 : atom_order
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
        type_order=None):
    """
    Converts ASE atoms to GPUMD compatible position file

    Parameters
    ----------
    arg1 : atoms
        Atoms to write to gpumd file

    arg2 : M
        Maximum number of neighbors for one atom

    arg3 : cutoff
        initial cutoff distance for building the neighbor list

    arg4 : gpumd_file
        file to save the structure data to

    arg5 : sort_key
        How to sort atoms ('group', 'type'). Default is None.

    arg6 : type_order
        List of atomic symbols in order to be listed in GPUMD xyz file. Default
        is None

    Returns
    -------

    """

    if sort_key == 'type':
        atoms_list = sorted(atoms, key=lambda x: __atoms_sortkey(x, atom_order))
    elif sort_key == 'group':
        atoms_list = sorted(atoms, key=lambda x: __atoms_sortkey(x))
    else:
        atoms_list = atoms

    if sort_key=='type' and type_order:
        types = type_order
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

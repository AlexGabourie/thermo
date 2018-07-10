from ase import Atom, Atoms
from math import floor

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def read_xyz(filename='xyz.in'):
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
    
    return atoms, M, cutoff


def read_traj(traj_file='xyz.out', in_file='xyz.in'):
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

    atoms_in, M, cutoff = read_xyz(in_file)
    N = len(atoms_in)

    num_frames = len(xyz_lines)/float(N)
    if not (num_frames == floor(num_frames)):
        print('read_traj warning: Non-integer number of frames base on number of atoms.' + 
              ' Only taking {} frames'.format(floor(num_frames)))

    num_frames = int(floor(num_frames))
    traj = list()
    for frame in range(num_frames):
        for i, line in enumerate(xyz_lines[frame*N:(frame+1)*N]):
            curr_atom = atoms_in[i]
            curr_atom.position = tuple([float(val) for val in line.split()])
        traj.append(atoms_in.copy())
    
    return traj
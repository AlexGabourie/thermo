from ase.io import write
from ase.io import read
from ase import Atom, Atoms
from readers import *

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

def convert_gpumd_xyz(in_file='xyz.in', out_filename='in.xyz', format='xyz'):
    """
    Converts the GPUMD input file to any compatible ASE output file
    
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
    atoms, M, cutoff = read_xyz(in_file)
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
    traj = read_traj(traj_file, in_file)
    write(out_filename, traj, format)
    return


def lammps2gpumd(filename, M, cutoff, style='atomic', gpumd_file='xyz.in'):
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
    atoms2gpumd(atoms, M, cutoff, gpumd_file=gpumd_file)
    return


def atoms2gpumd(atoms, M, cutoff, gpumd_file='xyz.in'):
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

    arg4 : style
        atom style used in LAMMPS data file

    arg5 : gpumd_file
        file to save the structure data to

    Returns
    -------

    """

    # Load atoms
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
        for atom in atoms[:-1]:
            type_ = [type_dict[atom.symbol], atom.tag, atom.mass] + list(atom.position)
            f.writelines(' '.join([str(val) for val in type_]) + '\n')
        # Last line
        atom = atoms[-1]
        type_ = [type_dict[atom.symbol], atom.tag, atom.mass] + list(atom.position)
        f.writelines(' '.join([str(val) for val in type_]))
    return


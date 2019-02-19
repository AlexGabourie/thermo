from ase import Atom, Atoms
from math import floor
import numpy as np
import os

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


#########################################
# Structure preprocessing
#########################################

def __get_group(split, pos, direction):
    '''
    Gets the group that an atom belongs to based on its position. Only works in
    one direction as it is used for NEMD.

    Args:
        split (list(float)):
            List of boundaries. First element should be lower boundary of
            sim. box in specified direction and the last the upper.

        position (float):
            Position of the atom

        direction (str):
            Which direction the split will work

    Returns:
        out (int):
            Group of atom

    '''
    if direction == 'x':
        d = pos[0]
    elif direction == 'y':
        d = pos[1]
    else:
        d = pos[2]
    errmsg = 'Out of bounds error: {}'.format(d)
    for i,val in enumerate(split[:-1]):
        if i == 0 and d < val:
            print errmsg
            return -1
        if d >= val and d < split[i+1]:
            return i
    print errmsg
    return -1

def __init_index(index, info, num_atoms):
    '''
    Initializes the index key for the info dict.

    Args:
        index (int):
            Index of atom in the Atoms object.

        info (dict):
            Dictionary that stores the velocity, layer, and groups.

        num_atoms (int):
            Number of atoms in the Atoms object.

    Returns:
        index (int)
            Index of atom in the Atoms object.

    '''
    if index == num_atoms-1:
        index = -1
    if not index in info.keys():
        info[index] = dict()
    return index

def __handle_end(info, num_atoms):
    '''
    Duplicates the index -1 entry for key that's num_atoms-1. Works in-place.

    Args:
        info (dict):
            Dictionary that stores the velocity, layer, and groups.

        num_atoms (int):
            Number of atoms in the Atoms object.

    '''
    info[num_atoms-1] = info[-1]

def add_group_by_position(split, atoms, direction):
    '''
    Assigns groups to all atoms based on its position. Only works in
    one direction as it is used for NEMD.
    Returns a bookkeeping parameter, but atoms will be udated in-place.

    Args:
        split (list(float)):
            List of boundaries. First element should be lower boundary of sim.
            box in specified direction and the last the upper.

        atoms (ase.Atoms):
            Atoms to group

        direction (str):
            Which direction the split will work.

    Returns:
        counts (int)
            A list of number of atoms in each group.

    '''
    info = atoms.info
    counts = [0]*(len(split)-1)
    num_atoms = len(atoms)
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        i = __get_group(split, atom.position, direction)
        if 'groups' in info[index].keys():
            info[index]['groups'].append(i)
        else:
            info[index]['groups'] = [i]
        counts[i] += 1
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts

def add_group_by_type(atoms, groups):
    '''
    Assigns groups to all atoms based on atom types. Returns a
    bookkeeping parameter, but atoms will be udated in-place.

    Args:
        atoms (ase.Atoms):
            Atoms to group

        types (dict):
            Dictionary with types for keys and group as a value.
            Only one group allowed per atom. Assumed groups are integers
            starting at 0 and increasing in steps of 1. Ex. range(0,10).

    Returns:
        counts (int)
            A list of number of atoms in each group.

    '''
    # atom symbol checking
    all_symbols = groups.keys()
    # check that symbol set matches symbol set of atoms
    if set(atoms.get_chemical_symbols()) - set(all_symbols):
        raise ValueError('Group symbols do not match atoms symbols.')
    if not len(set(all_symbols)) == len(all_symbols):
        raise ValueError('Group not assigned to all atom types.')

    num_groups = len(set([groups[sym] for sym in set(all_symbols)]))
    num_atoms = len(atoms)
    info = atoms.info
    counts = [0]*num_groups
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        group = groups[atom.symbol]
        counts[group] += 1
        if 'groups' in info[index].keys():
            info[index]['groups'].append(group)
        else:
            info[index]['groups'] = [group]
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts


def assign_layer_by_position(split, atoms, direction):
    '''
    Assigns layers to all atoms based on its position. Only works in
    one direction. Similar to group but only one layer can be assigned
    to an atom.
    Returns a bookkeeping parameter, but atoms will be udated in-place.

    Args:
        split (list(float)):
            List of boundaries. First element should be lower boundary of sim.
            box in specified direction and the last the upper.

        atoms (ase.Atoms):
            Atoms to assign layers to

        direction (str):
            Which direction the split will work

    Returns:
        counts (int)
            A list of number of atoms in each layer

    '''
    info = atoms.info
    num_atoms = len(atoms)
    counts = [0]*(len(split)-1)
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        i = __get_group(split, atom.position, direction)
        info[index]['layer'] = i
        counts[i] += 1
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts

def assign_layer_by_type(atoms, layers):
    '''
    Assigns a layer to all atoms based on atom types. Returns a
    bookkeeping parameter, but atoms will be udated in-place.

    Args:
        atoms (ase.Atoms):
            Atoms to assign layer to.

        types (dict):
            Dictionary with types for keys and layer as a value.
            Only one layer allowed per atom. Assumed layers are integers
            starting at 0 and increasing in steps of 1. Ex. range(0,10)

    Returns:
        counts (int)
            A list of number of atoms in each layer.

    '''
    # atom symbol checking
    all_symbols = layers.keys()
    # check that symbol set matches symbol set of atoms
    if set(atoms.get_chemical_symbols()) - set(all_symbols):
        raise ValueError('Layer symbols do not match atoms symbols.')
    if not len(set(all_symbols)) == len(all_symbols):
        raise ValueError('Layer not assigned to all atom types.')

    num_layers = len(set([layers[sym] for sym in set(all_symbols)]))
    num_atoms = len(atoms)
    info = atoms.info
    counts = [0]*num_layers
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        layer = layers[atom.symbol]
        counts[layer] += 1
        info[index]['layer'] = layer
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts

def set_velocities(atoms, custom=None):
    """
    Sets the 'velocity' part of the atoms to be used in GPUMD.
    Custom velocities must be provided. They must also be in
    the units of eV^(1/2) amu^(-1/2)

    Args:
        atoms (ase.Atoms):
            Atoms to assign velocities to.

        custom (list(list)):
             list of len(atoms) with each element made from
             a 3-element list for [vx, vy, vz]

    """
    if not custom:
        raise ValueError("No velocities provided.")

    num_atoms = len(atoms)
    info = atoms.info
    if not len(custom) == num_atoms:
        return ValueError('Incorrect number of velocities for number of atoms.')
    for index, (atom, velocity) in enumerate(zip(atoms, custom)):
        if not len(velocity) == 3:
            return ValueError('Three components of velocity not provided.')
        index = __init_index(index, info, num_atoms)
        info[index]['velocity'] = velocity
    __handle_end(info, num_atoms)
    atoms.info = info

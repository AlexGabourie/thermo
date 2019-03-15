import pickle

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"

###################################
# UFF
###################################

def load_UFF():
    """
    Loads dictionary that stores relevant LJ from UFF.

    Returns:
        out (dict):
            Dictionary with atom symbols as the key and a tuple of epsilon and
            sigma in units of eV and Angstroms, respectively.
    """
    
    return pickle.load(open('../../data/UFF.params', 'r'))

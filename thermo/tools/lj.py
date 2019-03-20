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

#################################
# LJ Object
#################################

class LJ(object):
    """ Stores all atoms for a simulation with their LJ parameters.

    A special dictionary with atom symbols for keys and the epsilon and
    sigma LJ parameters for the values. This object interfaces with the UFF
    LJ potential parameters but can also accept arbitrary parameters.
    """
    def __init__(self, symbols=None):
        """
        Args:
            symbols (str or list(str)):
                Optional input. A single symbol or a list of symbols to add to
                the initial LJ list.
        """
        self.ljdict = dict()
        if symbols:
            self.add_UFF_params(symbols)

    def add_UFF_params(self, symbols, replace=False):
        """
        Adds UFF parameters to the LJ object. Will replace existing parameters
        if 'replace' is set to True. UFF parameters are loaded from the package.

        Args:
            symbols (str or list(str)):
                A single symbol or a list of symbols to add to the initial LJ list.

            replace (bool):
                Whether or not to replace existing symbols
        """
        if type(symbols) == str:
            symbols = [symbols] # convert to list if one string
        UFF = load_UFF()
        for symbol in symbols:
            if not replace and symbol in self.ljdict.keys():
                print("Warning: {} is already in LJ list and will not be" + \
                "included.\n".format(symbol) + "To include, use " + \
                "replace_UFF_params or toggle 'replace' boolean.\n")
            else:
                self.ljdict[symbol] = UFF[symbol]

    def replace_UFF_params(self, symbols, add=False):
        """
        Replaces UFF parameters in the LJ object. Will add new entries if 'add'
        is set to True. UFF parameters are loaded from the package.

        Args:
            symbols (str or list(str)):
                A single symbol or a list of symbols to add to the initial LJ list.

            add (bool):
                Whether or not to replace existing symbols
        """
        if type(symbols) == str:
            symbols = [symbols] # convert to list if one string
        UFF = load_UFF()
        for symbol in symbols:
            if symbol in self.ljdict.keys() or add:
                self.ljdict[symbol] = UFF[symbol]
            else:
                print("Warning: {} is not in LJ list and will not be " +\
                "included.\n".format(symbol) + "To include, use " +\
                "add_UFF_params or toggle 'add' boolean.\n")


    def add_param(self, symbol, data, replace=True):
        """
        Adds a custom parameter to the LJ object.

        Args:
            symbol (str):
                Symbol of atom type to add.

            data (tuple(float)):
                A two-element tuple of numbers to represent the epsilon and
                sigma LJ values.

            replace (bool):
                Whether or not to replace the item.
        """

        # check params
        good = tuple == type(data) and len(data) == 2 and \
            all([isinstance(item, (int, long, float)) for item in data]) and \
            type(symbol) == str
        if good:
            if symbol in self.ljdict.keys():
                if replace:
                    self.ljdict[symbol] = data
                else:
                    print("Warning: {} exists and cannot be added.\n".format(symbol)
            else:
                selfljdict[symbol] = data

        else:
            raise ValueError("Invalid data parameter.")

    def remove_param(self, symbol):
        """
        Removes an element from the LJ object. If item does not exist, nothing
        happens.

        Args:
            symbol (str):
                Symbol of atom type to remove.
        """
        self.ljdict.pop(symbol, None)

    def __str__(self):
        out_str = 'Symbol: Epsilon (eV), Sigma (Angs.)\n'
        for key in self.ljdict.keys():
            cur = self.ljdict[key]
            out_str += "{}: {}, {}\n".format(key, cur[0], cur[1])
        return out_str

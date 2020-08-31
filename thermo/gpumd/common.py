import re
import os

__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def __get_direction(directions):
    """
    Creates a sorted list showing which directions the user asked for. Ex: 'xyz' -> ['x', 'y', 'z']

    Args:
        directions (str):
            A string containing the directions the user wants to process (Ex: 'xyz', 'zy', 'x')

    Returns:
        list(str): An ordered list that simplifies the user input for future processing

    """
    if not (bool(re.match('^[xyz]+$',directions))
            or len(directions) > 3
            or len(directions) == 0):
        raise ValueError('Invalid directions used.')
    return sorted(list(set(directions)))


def __get_path(directory, filename):
    if not directory:
        return os.path.join(os.getcwd(), filename)
    return os.path.join(directory, filename)


def __check_list(data, varname=None, dtype=None):
    """
    Checks if data is a list of dtype or turns a variable of dtype into a list

    Args:
        data

        varname (str):
            Name of variable to check

        dtype (type)
            Data type to check data against
    Returns:
        Bool
    """

    if type(data) == dtype:
        return [data]

    if type(data) == list:
        for elem in data:
            if not type(elem) == dtype:
                if varname:
                    raise ValueError('All entries for {} must be {}.'.format(str(varname), str(dtype)))
        return data

    raise ValueError('{} is not the correct type.'.format(str(varname)))
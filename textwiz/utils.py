import os
import json
import yaml
import random
from typing import Callable, TypeVar, ParamSpec

import torch
import numpy as np


P = ParamSpec("P")
T = TypeVar("T")

# Path to the root of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

# Path to the data folder
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')


# Most frequent text/data file extensions
FREQUENT_EXTENSIONS = (
    'json',
    'jsonl',
    'txt',
    'csv'
)


def set_all_seeds(seed: int):
    """Set seed for all random number generators (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def load_json(filename: str) -> dict:
    """
    Load a json file and return a dictionary.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    data : dict
        The dictionary representing the file.

    """
    
    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data


def load_yaml(filename: str) -> dict:
    """Load a yaml file as a dict.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    dict
        The output.
    """

    with open(filename, 'r') as fp:
        data = yaml.safe_load(fp)

    return data


def validate_filename(filename: str, extension: str = 'json') -> str:
    """Format and check the validity of a filename and its extension. Create the path if needed, and 
    add/manipulate the extension if needed.

    Parameters
    ----------
    filename : str
        The filename to check for.
    extension : str, optional
        The required extension for the filename, by default 'json'

    Returns
    -------
    str
        The filename, reformated if needed.
    """

    # Extensions are always lowercase
    extension = extension.lower()
    # Remove dots in extension if any
    extension = extension.replace('.', '')

    dirname, basename = os.path.split(filename)

    # Check that the extension and basename are correct
    if basename == '':
        raise ValueError('The basename cannot be empty')
    
    split_on_dots = basename.split('.')

    # In this case add the extension at the end
    if len(split_on_dots) == 1:
        basename += '.' + extension
    # In this case there may be an extension, and we check that it is the correct one and change it if needed
    else:
        # The extension is correct
        if split_on_dots[-1] == extension:
            pass
        # There is a frequent extension, but not the correct one -> we change it
        elif split_on_dots[-1] in FREQUENT_EXTENSIONS:
            basename = '.'.join(split_on_dots[0:-1]) + '.' + extension
        # We did not detect any extension -> just add it at the end
        else:
            basename = '.'.join(split_on_dots) + '.' + extension

    # Check that the given path goes through the project repository
    dirname = os.path.abspath(dirname)
    if not (dirname.startswith(ROOT_FOLDER + os.sep) or dirname == ROOT_FOLDER):
        raise ValueError('The path you provided is outside the project repository.')

    # Make sure the path exists, and creates it if this is not the case
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return os.path.join(dirname, basename)


def save_json(dictionary: dict, filename: str):
    """
    Save a dictionary to disk as a json file.

    Parameters
    ----------
    dictionary : dict
        The dictionary to save.
    filename : str
        Filename to save the file.
    """
    
    filename = validate_filename(filename, extension='json')
    
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')


def copy_docstring_and_signature(copied_func: Callable[P, T]):
    """Decorator that copies the docstring and signature of another function.
    Note: the type hints are absolutely necessary for VScode to properly show the signature and docstring
    of the new function. There is some black magic in how VScode shows the docstrings because simply
    updating the __doc__ property does not work...
    
    Parameters
    ----------
    copied_func : Callable[P, T]
        The function from which to copy docstring and signature.
    """

    def wrapper(original_func: Callable[P, T]) -> Callable[P, T]:
        original_func.__doc__ = copied_func.__doc__
        return original_func
    
    return wrapper

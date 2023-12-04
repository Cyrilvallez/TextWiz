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

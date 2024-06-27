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


def memory_estimation_causal(reference_file: str, input_size: int, max_new_tokens: int) -> tuple[float, bool]:
    """Compute the memory needed in GiB for a batch size of 1 given the current `reference_file` (memory estimations for
    a given causal model), and current `input_size` and `max_new_tokens`.

    Parameters
    ----------
    reference_file : str
        File containing the memory estimations for a given model and dtype.
    input_size : int
        The input length.
    max_new_tokens : int
        The number of tokens to generate.
    Returns
    -------
    tuple[float, bool]
        Tuple containing the memory needed in GiB and whether this estimation is good or not (depending on goodness
        of fit).
    """
    
    memory_footprints = load_json(reference_file)
    # Convert keys to int
    memory_footprints = {k1: {int(k2): v2 for k2, v2 in v1.items()} for k1, v1 in memory_footprints.items()}

    passes_r2_test = True
    fit_results = {}

    # Fit the curves
    for key in memory_footprints.keys():
        x = np.array(list(memory_footprints[key].keys()))
        y = np.array(list(memory_footprints[key].values()))
        # Make sure vectors are sorted correctly (dics should stay ordered but you never know)
        sorting = np.argsort(x)
        x = x[sorting]
        y = y[sorting]

        # Memory usage of forward pass without cache is linear when using flash attention implementation, else quadratic.
        if key == 'without cache':
            # First try linear
            fit, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)
            r2 = r_squared(stats[0], y)
            # If bad fit, fallback to quadratic
            if r2 < 0.95:
                fit, stats = np.polynomial.Polynomial.fit(x, y, deg=2, full=True)
        # Memory usage of cache and forward pass using cache is always linear.
        else:
            fit, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)

        r2 = r_squared(stats[0], y)
        # This should always be the case, but check it if for some reason the behavior is not sufficiently linear (or quadratic)
        if r2 < 0.95:
            passes_r2_test = False
        fit_results[key] = fit


    memory_needed_without_cache = fit_results['without cache'](input_size)
    memory_needed_with_cache = fit_results['cache size'](input_size + max_new_tokens) + fit_results['with cache'](input_size + max_new_tokens)
    memory_needed = max(memory_needed_without_cache, memory_needed_with_cache)

    return memory_needed, passes_r2_test


def memory_estimation_embedding(reference_file: str, input_size: int) -> float:
    """Compute the memory needed in GiB for a batch size of 1 given the current `reference_file` (memory estimations for
    a given embedding model), and current `input_size`.

    Parameters
    ----------
    reference_file : str
        File containing the memory estimations for a given model and dtype.
    input_size : int
        The input length.
    Returns
    -------
    tuple[float, bool]
        Tuple containing the memory needed in GiB and whether this estimation is good or not (depending on goodness
        of fit).
    """
    
    memory_footprints = load_json(reference_file)
    # Convert keys to int
    memory_footprints = {int(k): v for k, v in memory_footprints.items()}

    passes_r2_test = True

    # Fit the curve
    x = np.array(list(memory_footprints.keys()))
    y = np.array(list(memory_footprints.values()))
    # Make sure vectors are sorted correctly (dics should stay ordered but you never know)
    sorting = np.argsort(x)
    x = x[sorting]
    y = y[sorting]

    # Memory usage of forward pass is linear when using flash attention implementation, else quadratic.
    # First try linear
    fit, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)
    r2 = r_squared(stats[0], y)
    # If bad fit, fallback to quadratic
    if r2 < 0.95:
        fit, stats = np.polynomial.Polynomial.fit(x, y, deg=2, full=True)
        r2 = r_squared(stats[0], y)
        # This should always be the case, but check it if for some reason the behavior is not sufficiently linear (or quadratic)
        if r2 < 0.95:
            passes_r2_test = False

    memory_needed = fit(input_size)

    return memory_needed, passes_r2_test


def r_squared(residual: float, y: np.array) -> float:
    """Compute the coefficient of determination (R^2) of a numpy fit."""
    SS_tot = sum((y - np.mean(y))**2)
    return 1 - residual / SS_tot
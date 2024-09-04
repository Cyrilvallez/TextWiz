import os
import argparse

from . import loader
from .helpers import utils


ALLOWED_DTYPES = ('int4', 'int8', 'float16', 'bfloat16', 'float32', 'float64')


def memory_estimation_entrypoint() -> float:
    """Compute the memory needed in GiB for a batch size of 1 given the current `model_name`, `dtype`, and generation
    parameters.

    Returns
    -------
    float
        The memory needed for inference.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Inference memory estimation')
    parser.add_argument('model_name', type=str, choices=loader.ALLOWED_MODELS,
                        help='The model to use.')
    parser.add_argument('input_size', type=int,
                        help='Number of tokens of the input.')
    parser.add_argument('new_tokens', type=int,
                        help='Number of new tokens to generate (pass 0 if embedding model)')
    parser.add_argument('--dtype', type=str, choices=ALLOWED_DTYPES,
                        help='The dtype of the model.')
    args = parser.parse_args()
    model_name = args.model_name
    input_size = args.input_size
    max_new_tokens = args.new_tokens
    dtype = args.dtype


    loader.check_model_name(model_name)
    if dtype is None:
        dtype = str(loader.get_model_dtype(model_name)).split('.', 1)[1]
    if dtype not in ALLOWED_DTYPES:
        raise ValueError('The dtype you provided is not valid.')
    
    is_embedding_model = model_name in loader.ALLOWED_EMBEDDING_MODELS
    version_ = "old" if utils._is_old_version else "new"
    model_category = 'embedding' if is_embedding_model else 'causal'
    reference_file = os.path.join(utils.DATA_FOLDER, 'memory_estimator', version_, model_category, model_name, f'{dtype}.json')

    if not os.path.exists(reference_file):
        raise ValueError('No memory estimation currently exists for the given combination of `model_name` and `dtype`.')

    if is_embedding_model:
        memory, r2_test = utils.memory_estimation_embedding(reference_file, input_size)
    else:
        memory, r2_test = utils.memory_estimation_causal(reference_file, input_size, max_new_tokens)

    if not r2_test:
        raise ValueError(f'Memory estimation cannot be precisely computed (behavior not sufficiently linear/quadratic).')
    
    return memory

    
import os
import time
import argparse
import logging
import importlib.metadata
from packaging import version

import torch
import numpy as np
from transformers import DynamicCache

from textwiz import HFCausalModel, HFEmbeddingModel, loader, dtype_category
from textwiz.helpers import warnings_suppressor, utils
from textwiz.helpers.constants import RANDOM_LONG_TEXT

# Remove warning when tokenizing sequences longer than expected: we know we are doing it!
logger = logging.getLogger('transformers.tokenization_utils_base')
logger.addFilter(warnings_suppressor.LoggingFilter("Token indices sequence length is longer than the specified maximum sequence length for this model"))

_is_old_version = utils._is_old_version


def memory_usage(past_key_values):
    """Recursively compute the memory footprint of past key values (in bytes).
    """

    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.nelement() * past_key_values.element_size()
    elif isinstance(past_key_values[0], torch.Tensor):
        return sum([x.nelement() * x.element_size() for x in past_key_values])
    else:
        return sum([memory_usage(x) for x in past_key_values])


def single_memory_pass_causal(model: HFCausalModel, input_ids: torch.Tensor) -> tuple[float, float, float]:
    """Returns the memory usage of the forward pass creating the cache, memory usage of the cache, and memory usage of
    a second forward pass using the cache for the given `input_ids`. Everything in GiB.
    Note that this is handy to wrap in a function as the potentially big outputs (cache) are immediately 
    deallocated after the call.

    Parameters
    ----------
    model : HFCausalModel
        The model to benchmark.
    input_ids : torch.Tensor
        The input tokens.

    Returns
    -------
    tuple[float, float, float]
        Memory usages in GiB.
    """

    gpus = model.get_gpu_devices()

    actual_peaks = {}
    for gpu_rank in gpus:
        torch.cuda.reset_peak_memory_stats(gpu_rank)
        actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

    # Single forward pass creating the K-V cache
    with torch.no_grad():
        # Initialize a DynamicCache as will be done by default
        past_key_values = DynamicCache() if model.model._supports_default_dynamic_cache() else None
        try:
            _supports_logits = model.model._supports_num_logits_to_keep()
        except AttributeError:
            _supports_logits = False
        if _supports_logits:
            output = model.model(input_ids, past_key_values=past_key_values, return_dict=True, use_cache=True,
                                 num_logits_to_keep=1)
        else:
            output = model.model(input_ids, past_key_values=past_key_values, return_dict=True, use_cache=True)
    
    memory_used = {}
    for gpu_rank in gpus:
        memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
    
    # Actual largest memory usage peak accross gpus
    max_peak_without_cache = max(memory_used.values())
    cache_size = memory_usage(output.past_key_values) / 1024**3

    # Random new token to append
    new_token = torch.tensor([[124]], device=input_ids.device)
    # Formating inputs for second forward using cache
    new_input_ids = torch.cat((input_ids, new_token), dim=1)
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.int64, device=input_ids.device)
    input_dict = model.model.prepare_inputs_for_generation(new_input_ids, cache_position=cache_position, past_key_values=output.past_key_values,
                                                           use_cache=True)
    
    actual_peaks = {}
    for gpu_rank in gpus:
        torch.cuda.reset_peak_memory_stats(gpu_rank)
        actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3
    
    # New forward pass using the K-V cache
    with torch.no_grad():
        foo = model.model(**input_dict)

    memory_used = {}
    for gpu_rank in gpus:
        memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
    
    # Actual largest memory usage peak accross gpus
    max_peak_with_cache = max(memory_used.values())
    
    return max_peak_without_cache, cache_size, max_peak_with_cache


def single_memory_pass_embedding(model: HFEmbeddingModel, input_ids: torch.Tensor) -> float:
    """Returns the memory usage of the forward pass for embedding models. Everything in GiB.

    Parameters
    ----------
    model : HFEmbeddingModel
        The model to benchmark.
    input_ids : torch.Tensor
        The input tokens.

    Returns
    -------
    float
        Memory usages in GiB.
    """

    gpus = model.get_gpu_devices()

    actual_peaks = {}
    for gpu_rank in gpus:
        torch.cuda.reset_peak_memory_stats(gpu_rank)
        actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

    # Single forward pass
    with torch.no_grad():
        foo = model.model(input_ids)
    
    memory_used = {}
    for gpu_rank in gpus:
        memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
    
    # Actual largest memory usage peak accross gpus
    max_peak = max(memory_used.values())
    
    return max_peak


def memory_estimation(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                      max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8):
    """Estimate the memory needed to generate text depending on the context size. This function will also check
    if the memory scale with the full context (input size + max_new_tokens), or only with the input size. Indeed,
    in the first forward pass we do not already have the K-V cache, so it needs to be computed. However, in some
    cases the size of the cache is very small compared to the memory needed to compute it the first time, in which
    case the memory only scales with the memory footprint of the first forward pass.
    For embedding models, this function only computes the memory footprint of the forward.

    Parameters
    ----------
    model_name : str
        The name of the model.
    quantization_8bits : bool, optional
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool, optional
        Whether the model will be loaded in 4 bits mode, by default False.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.
    """

    t0 = time.time()

    # Override quantization for bloom due to its size
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        quantization_8bits = True

    CAUSAL = model_name in loader.ALLOWED_CAUSAL_MODELS
    EMBEDDING = model_name in loader.ALLOWED_EMBEDDING_MODELS

    # Initialize filenames and return if files already exist
    dtype_name = dtype_category(model_name, quantization_4bits=quantization_4bits, quantization_8bits=quantization_8bits)

    already_exist = False
    version_ = "old" if _is_old_version else "new"
    if CAUSAL:
        filename_memory = os.path.join(utils.DATA_FOLDER, 'memory_estimator', version_, 'causal', model_name, f'{dtype_name}.json')
        if os.path.exists(filename_memory):
            already_exist = len(utils.load_json(filename_memory)['without cache'].keys()) == 50
    elif EMBEDDING:
        filename_memory = os.path.join(utils.DATA_FOLDER, 'memory_estimator', version_, 'embedding', model_name, f'{dtype_name}.json')
        if os.path.exists(filename_memory):
            already_exist = len(utils.load_json(filename_memory).keys()) == 50
        
    # Immediately return if it already exist
    if already_exist:
        print(f'It seems like a memory estimation already exists for {model_name} and currently selected dtype.')
        return
    
    print(f'Starting with {model_name}!')

    # Load model
    if CAUSAL:
        model = HFCausalModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits,
                            max_fraction_gpu_0=max_fraction_gpu_0, max_fraction_gpus=max_fraction_gpus)
        model_memory_consumption = {'without cache': {}, 'with cache': {}, 'cache size': {}}
    elif EMBEDDING:
        model = HFEmbeddingModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits,
                                 max_fraction_gpu_0=max_fraction_gpu_0, max_fraction_gpus=max_fraction_gpus)
        model_memory_consumption = {}

    large_tokens = model.tokenizer.encode(RANDOM_LONG_TEXT*10, return_tensors='pt').to(model.input_device)

    # We go until 8192 tokens maximum for this benchmark (more tokens are rarely used)
    max_input_size = min(model.get_context_size(), 8192)
    # select input sizes to use depending on model max context
    input_sizes = np.linspace(32, max_input_size - 32, num=50, endpoint=True, dtype=int).tolist()

    # for input_size in tqdm(input_sizes, desc=model_name, file=sys.stdout):
    for input_size in input_sizes:

        # Select inputs
        input_ids = large_tokens[:, :input_size]
                
        # Try to get memory needs for current input_ids
        try:
            if CAUSAL:
                max_peak_without_cache, cache_size, max_peak_with_cache = single_memory_pass_causal(model, input_ids)
            elif EMBEDDING:
                max_peak = single_memory_pass_embedding(model, input_ids)
        # If OOM, exit loop and save results
        except RuntimeError as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                break
            else:
                raise e

        # Add entries to result dictionary
        if CAUSAL:
            model_memory_consumption['without cache'][input_size] = max_peak_without_cache
            model_memory_consumption['cache size'][input_size] = cache_size
            model_memory_consumption['with cache'][input_size + 1] = max_peak_with_cache
        elif EMBEDDING:
            model_memory_consumption[input_size] = max_peak
      
        # Save results
        utils.save_json(model_memory_consumption, filename_memory)

    dt = time.time() - t0

    print(f'Done with {model_name} in {dt/3600:.2f} h!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Memory estimator')
    parser.add_argument('model', type=str, choices=loader.ALLOWED_MODELS,
                        help='The model to use for memory estimation.')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--max_gpu_0', type=float, default=0.8,
                        help='Max fraction of gpu 0 memory to reserve for the model.')
    parser.add_argument('--max_gpus', type=float, default=0.8,
                        help='Max fraction of gpus (other than indice 0) memory to reserve for the model.')
    
    args = parser.parse_args()
    model = args.model
    int8 = args.int8
    int4 = args.int4
    max_gpu_0 = args.max_gpu_0
    max_gpus = args.max_gpus

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')
    
    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")

    # Perform the estimation
    memory_estimation(model, int8, int4, max_gpu_0, max_gpus)

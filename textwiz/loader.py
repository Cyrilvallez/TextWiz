import warnings
import re
import math
import importlib.metadata
from packaging import version

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from .configs import (
    ALL_MODELS_MAPPING,
    ALL_MODELS_DTYPES,
    ALL_MODELS_PARAMS,
    ALL_MODELS_FAMILY,
    ALL_MODELS_CONTEXT_SIZE,
    ALL_MODELS_VERSIONS,
    ALL_MODELS_ADDITIONAL_MODEL_KWARGS,
    ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS,
    ALL_MODELS_PURPOSE,
)


# Summarize all supported model names
ALLOWED_MODELS = tuple(ALL_MODELS_MAPPING.keys())
ALLOWED_CAUSAL_MODELS = tuple(model for model in ALL_MODELS_MAPPING.keys() if ALL_MODELS_PURPOSE[model] == 'causal')
ALLOWED_EMBEDDING_MODELS = tuple(model for model in ALL_MODELS_MAPPING.keys() if ALL_MODELS_PURPOSE[model] == 'embedding')

ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Mapping between purpose and transformer class
BASE_MODEL_CLASS_MAPPING = {
    'causal': AutoModelForCausalLM,
    'embedding': AutoModel,
}


def get_model_params(model_name: str) -> float:
    """Return the approximate number of params of the model, in billions.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    float
        The number of parameters.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_PARAMS[model_name]


def get_model_dtype(model_name: str) -> torch.dtype:
    """Return the default dtype used by the model.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    torch.dtype
        The default dtype.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_DTYPES[model_name]


def get_model_context_size(model_name: str) -> int:
    """Return the maximum context size used by the model.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    int
        The context size.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_CONTEXT_SIZE[model_name]



def check_versions(model_name: str):
    """Ensure that the versions are compatible with the current `model_name`, and raises an error if it is 
    not the case.

    Parameters
    ----------
    model_name : str
        The model name.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    if model_name not in ALL_MODELS_VERSIONS.keys():
        return

    versions = ALL_MODELS_VERSIONS[model_name]
    packages_to_check = list(versions.keys())

    pattern = r'(>=|<=)([0-9]+\.[0-9]+\.[0-9]+)'
    comparators = {'>=': lambda x, y: x >= y, '<=': lambda x, y: x <= y}
    
    version_errors = []
    for package in packages_to_check:

        # Check package version without importing
        actual_version = version.parse(importlib.metadata.version(package))
        requirements = versions[package]
        parsing = re.findall(pattern, requirements)

        for comparator, required_version in parsing:
            required_version = version.parse(required_version)
            if not comparators[comparator](actual_version, required_version):
                version_errors.append(f'{package} {comparator} {required_version}')

    if len(version_errors) > 0:
        raise RuntimeError(f'{model_name} requires the following package versions: {*version_errors,}')
    else:
        return



def estimate_model_gpu_footprint(model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                                 max_fraction_gpus: float = 0.8) -> tuple[int, dict]:
    """Estimate the minimum number of gpus needed to perform inference with a model, given the maximum gpu memory
    proportion `max_fraction_gpu_0` and `max_fraction_gpus` that we allow for the model. This relies on
    simple heuristics. This also computes the corresponding `memory_map` to use when creating a `device_map`.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    tuple[int, dict]
        Tuple containing the minimum number of gpus needed, the `memory_map`, i.e. a dictionary mapping each gpu
        needed to the maximum size reserved by the model for this gpu.
    """

    if max_fraction_gpu_0 < 0 or max_fraction_gpus < 0:
        raise ValueError('The maximum fraction of gpu memory to use cannot be negative.')
    
    if max_fraction_gpu_0 > 0.95 or max_fraction_gpus > 0.95:
        raise ValueError(('The maximum fraction of gpu memory to use cannot be larger than 0.95 because some '
                         'memory need to stay free for the forward pass and other computations.'))
    
    # Silently use 4bits when both are True
    if quantization_4bits and quantization_8bits:
        quantization_8bits = False

    # In this case we set it to 0.85 because otherwise bitsandbytes complain that we don't have enough resources
    # but in practice after loading the model uses less memory than this
    if (model_name == 'bloom-176B' and quantization_8bits and not quantization_4bits and \
        max_fraction_gpu_0 == 0.8 and max_fraction_gpus == 0.8):
        max_fraction_gpu_0 = 0.85
        max_fraction_gpus = 0.85

    # If not provided take the default one
    if dtype is None:
        dtype = get_model_dtype(model_name)

    if quantization_4bits:
        size_multiplier = 1/2
    elif quantization_8bits:
        size_multiplier = 1
    elif (dtype == torch.float16) or (dtype == torch.bfloat16):
        size_multiplier = 2
    else:
        size_multiplier = 4

    # Estimate of the memory size of the model
    rough_model_size_estimate = ALL_MODELS_PARAMS[model_name] * size_multiplier
    
    # We assume that we always have identical gpus when using multiple gpus
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_memory = 40
        warnings.warn('Could not find any GPUs on your system. Showing estimation for a system equipped with A100 40GB GPUs.')
    # Say we only have access to a portion of that memory for our model
    gpu_0_available_memory = max_fraction_gpu_0 * gpu_memory
    gpus_available_memory = max_fraction_gpus * gpu_memory

    # Heuristic: if the remainder is smaller than 2% of gpu_memory, do not add a gpu
    if rough_model_size_estimate <= gpu_0_available_memory + 0.02 * gpu_memory:
        return 1, None
    
    else:
        max_memory_map = {0: f'{math.ceil(gpu_0_available_memory)}GiB'}
        to_fit_on_other_gpus = rough_model_size_estimate - gpu_0_available_memory
        additional_gpus_needed = int(to_fit_on_other_gpus // gpus_available_memory)
        # This is bound to be (almost) always True. For multiple GPU setup, we do not try to distill the remainder
        # between GPUs even if it is small.
        if to_fit_on_other_gpus % gpus_available_memory > 0:
            additional_gpus_needed += 1
        available_gpu_size = math.ceil(gpus_available_memory)

        gpu_needed = 1 + additional_gpus_needed
        for i in range(1, gpu_needed):
            max_memory_map[i] = f'{available_gpu_size}GiB'

        return gpu_needed, max_memory_map



def load_model(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
               dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
               device_map: dict | str | None = None, gpu_rank: int = 0):
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
        The model.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    # Check package versions
    check_versions(model_name)
    
    # Set the dtype if not provided
    if dtype is None:
        dtype = ALL_MODELS_DTYPES[model_name]

    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f'The dtype must be one of {*ALLOWED_DTYPES,}.')
    
    if quantization_8bits and quantization_4bits:
        raise ValueError(('You cannot load a model with both `quantization_8bits` and `quantization_4bits`. '
                         'Please choose one'))
    
    # torch.float16 is not supported on cpu
    if not torch.cuda.is_available() and dtype != torch.float32:
        dtype = torch.float32
    
    # Override quantization if we don't have access to GPUs
    if not torch.cuda.is_available() and (quantization_8bits or quantization_4bits):
        quantization_4bits = False
        quantization_8bits = False
        warnings.warn('There are no GPUs available. The model will NOT be quantized.', RuntimeWarning)

    # Flag to know if the model is quantized
    quantization = quantization_8bits or quantization_4bits

    # Override dtype if we quantize the model as only float16 is acceptable for quantization
    dtype = torch.float16 if quantization else dtype

    # Add possible additional kwargs
    if model_name in ALL_MODELS_ADDITIONAL_MODEL_KWARGS.keys():
        additional_kwargs = ALL_MODELS_ADDITIONAL_MODEL_KWARGS[model_name]
    else:
        additional_kwargs = {}


    # Flag that will be set to True if we don't even need a device_map and can just put the model on one gpu
    only_move_to_one_gpu = False
    
    # Automatically find the best device_map depending on the model size and gpu size.
    # Try to minimize the number of gpus to use because using more will slow inference (but allow larger
    # batch size -> hard trade-off to find). Indeed, the parallelism of device_map is naive and gpus are only
    # used sequentially
    if (device_map is None) and torch.cuda.is_available():
    
        min_gpu_needed, max_memory_map = estimate_model_gpu_footprint(model_name, quantization_8bits=quantization_8bits,
                                                                      quantization_4bits=quantization_4bits, dtype=dtype,
                                                                      max_fraction_gpu_0=max_fraction_gpu_0,
                                                                      max_fraction_gpus=max_fraction_gpus)
        gpu_number = torch.cuda.device_count()

        if min_gpu_needed > gpu_number:
            raise RuntimeError(("The model seems too big for the gpu resources you have. To offload to the cpu as well, "
                               "explicitly pass a `device_map`, e.g. device_map='balanced'."))
        
        # In this case we don't need a device_map, we just move the model to the 1st gpu. Most models are 
        # relatively small and should fall on this category.
        if min_gpu_needed == 1:
            only_move_to_one_gpu = True
            # This is needed to move the model to the correct gpu when using quantization
            if quantization:
                device_map = {'': gpu_rank}
        # In this case, we need more than 1 gpu so we create a device_map between different gpus. However, 
        # we minimize the number of gpus used with the max_memory arg instead of naively using device_map='balanced'
        # between all gpus, because the parallelism is not optimized and thus using a lot of gpus is not efficient
        # if not needed
        else:
            additional_kwargs['max_memory'] = max_memory_map
            # Providing 'balanced' dispatch correctly with respect to the max_memory_map we provide
            device_map = 'balanced'

    # Base class for loading
    base_class = BASE_MODEL_CLASS_MAPPING[ALL_MODELS_PURPOSE[model_name]]

    # Load model
    # We first try with flash attention 2
    try:
        model = base_class.from_pretrained(ALL_MODELS_MAPPING[model_name], attn_implementation='flash_attention_2',
                                           device_map=device_map, torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                           load_in_4bit=quantization_4bits, low_cpu_mem_usage=True, **additional_kwargs)
        success = True
    except:
        success = False
    
    # Second try with Pytorch native sdpa (which may sometimes but not for all models also use flash attention 2)
    if not success:
        try:
            model = base_class.from_pretrained(ALL_MODELS_MAPPING[model_name], attn_implementation='sdpa',
                                               device_map=device_map, torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                               load_in_4bit=quantization_4bits, low_cpu_mem_usage=True, **additional_kwargs)
            success = True
        except:
            success = False

    # Last try with BetterTransformer, which is the same as sdpa but with coverage for more models
    if not success:
        model = base_class.from_pretrained(ALL_MODELS_MAPPING[model_name], attn_implementation='eager', device_map=device_map,
                                           torch_dtype=dtype, load_in_8bit=quantization_8bits, load_in_4bit=quantization_4bits,
                                           low_cpu_mem_usage=True, **additional_kwargs)
        # For some reason bettertransformer is supported for codegen2 models but makes them crash during the forward
        if not ('codegen2-' in model_name):
            # Convert to better transformer to use Pytorch optimizations if supported by the model
            try:
                model = model.to_bettertransformer()
            except:
                warnings.warn(('The default manual attention implementation will be used. This will result in slower generation and '
                               'higher memory usage. This should not be an issue for small models.'))
        
    
    # If the flag is active we directly put our model on one gpu without using any device_map (this is 
    # more efficient). But if the model is quantized, this is already done automatically because quantization
    # happen only on gpu
    if only_move_to_one_gpu and not quantization:
        # This operation is in-place for nn.Module
        model.cuda(gpu_rank)
        
    model.eval()

    return model


def load_tokenizer(model_name: str):
    """Load a pretrained tokenizer corresponding to one of the supported models.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
        The tokenizer.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.') 
    
    # Check package versions
    check_versions(model_name)
    
    if model_name in ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS.keys():
        additional_kwargs = ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS[model_name]
    else:
        additional_kwargs = {}
    
    tokenizer = AutoTokenizer.from_pretrained(ALL_MODELS_MAPPING[model_name], **additional_kwargs)

    return tokenizer


def load_model_and_tokenizer(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                             dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                             max_fraction_gpus: float = 0.8, device_map: dict | str | None = None,
                             gpu_rank: int = 0) -> tuple:
    """Load both a model and corresponding tokenizer.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
    tuple
        The model and tokenizer.
    """

    return (load_model(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits,
                       dtype=dtype, max_fraction_gpu_0=max_fraction_gpu_0, max_fraction_gpus=max_fraction_gpus,
                       device_map=device_map, gpu_rank=gpu_rank),
            load_tokenizer(model_name))


import warnings
import re
import math
import importlib.metadata
from packaging import version

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _infer_model_size(model_name: str) -> float:
    """Return the number of parameters a model has from its name if it can be inferred from it. Raise a 
    ValueError otherwise.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    float
        The number of parameters of the model, in billions.
    """

    # The following regex matches any digits possibly separated with a dot ('.') which is immeditely
    # followed by a 'B' or 'M' to capture the model size following our model name convention. Parenthesis 
    # allow to capture given groups of the regex thanks to the match object .group() method.
    pattern = r'([0-9]+(?:\.[0-9]+)?)([BM])'

    match = re.search(pattern, model_name)
    if match:
        matched_number = match.group(1)
        matched_letter = match.group(2)
        # Model size in billion (B) of parameters
        model_size = float(matched_number) if matched_letter == 'B' else float(matched_number)/1e3
        return model_size
    else:
        raise ValueError('The model number of parameters cannot be inferred from its name.')
    

def _infer_model_sizes(name_mapping: dict[str, str]) -> dict[str, float]:
    """Infer the number of parameters of all model names (dict keys) and return them as {key: #params}.

    Parameters
    ----------
    name_mapping : dict[str, str]
        A dictionary whose keys are the model names.

    Returns
    -------
    dict[str, float]
        A mapping from names to number of parameters.
    """

    return {key: _infer_model_size(key) for key in name_mapping.keys()}


def _register_model(model_name: str):
    """Register a model into the global variables containing all models parameters.

    Parameters
    ----------
    model_name : str
        Name (prefix in uppercase) of the model.
    """

    # Protect against injections (any character except digits, uppercases and underscore are detected)
    if re.search(r'[^A-Z0-9_]', model_name):
        raise ValueError('Cannot register a model with a name containing special or lowercase characters.')

    required_suffixes = ('MODELS_MAPPING', 'MODELS_DTYPES', 'MODELS_PARAMS', 'MODELS_FAMILY', 'MODELS_CONTEXT_SIZE')
    optional_suffixes = ('MODELS_VERSIONS', 'MODELS_ADDITIONAL_MODEL_KWARGS', 'MODELS_ADDITIONAL_TOKENIZER_KWARGS')

    # Template string to `exec` in order to merge the dictionaries
    template = 'ALL_{suffix}.update({model_name}_{suffix})'

    for suffix in required_suffixes:
        code = template.format(suffix=suffix, model_name=model_name)
        exec(code)

    for suffix in optional_suffixes:
        try:
            code = template.format(suffix=suffix, model_name=model_name)
            exec(code)
        # In this case the optional variable is not present
        except NameError:
            pass


    
# Those will be updated with each call to _register_model()
ALL_MODELS_MAPPING = {}
ALL_MODELS_DTYPES = {}
ALL_MODELS_PARAMS = {}
ALL_MODELS_FAMILY = {}
ALL_MODELS_CONTEXT_SIZE = {}
ALL_MODELS_VERSIONS = {}
ALL_MODELS_ADDITIONAL_MODEL_KWARGS = {}
ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {}
    

# Pretrained bloom models
BLOOM_MODELS_MAPPING = {
    'bloom-560M': 'bigscience/bloom-560m',
    'bloom-1.7B': 'bigscience/bloom-1b7',
    'bloom-3B': 'bigscience/bloom-3b',
    'bloom-7.1B':'bigscience/bloom-7b1',
    'bloom-176B': 'bigscience/bloom',
}
BLOOM_MODELS_DTYPES = {
    'bloom-560M': torch.float16,
    'bloom-1.7B': torch.float16,
    'bloom-3B': torch.float16,
    'bloom-7.1B':torch.float16,
    'bloom-176B': torch.bfloat16,
}
BLOOM_MODELS_PARAMS = _infer_model_sizes(BLOOM_MODELS_MAPPING)
BLOOM_MODELS_FAMILY = {model: 'bloom' for model in BLOOM_MODELS_MAPPING.keys()}
BLOOM_MODELS_CONTEXT_SIZE = {model: 2048 for model in BLOOM_MODELS_MAPPING.keys()}
_register_model('BLOOM')


# Pretrained Dialo-GPT models
DIALO_GPT_MODELS_MAPPING = {
    'dialo-gpt-small': 'microsoft/DialoGPT-small',
    'dialo-gpt-medium': 'microsoft/DialoGPT-medium',
    'dialo-gpt-large': 'microsoft/DialoGPT-large',
}
DIALO_GPT_MODELS_DTYPES = {model: torch.float32 for model in DIALO_GPT_MODELS_MAPPING}
DIALO_GPT_MODELS_PARAMS = {
    'dialo-gpt-small': 125/1e3,
    'dialo-gpt-medium': 355/1e3,
    'dialo-gpt-large': 775/1e3,
}
DIALO_GPT_MODELS_FAMILY = {model: 'dialo-gpt' for model in DIALO_GPT_MODELS_MAPPING.keys()}
DIALO_GPT_MODELS_CONTEXT_SIZE = {model: 1024 for model in DIALO_GPT_MODELS_MAPPING.keys()}
_register_model('DIALO_GPT')


# Pretrained StableLM models
STABLE_LM_MODELS_MAPPING = {
    'stable-lm-3B': 'stabilityai/stablelm-base-alpha-3b',
    'stable-lm-7B': 'stabilityai/stablelm-base-alpha-7b',
}
STABLE_LM_MODELS_DTYPES = {model: torch.float16 for model in STABLE_LM_MODELS_MAPPING.keys()}
STABLE_LM_MODELS_PARAMS = _infer_model_sizes(STABLE_LM_MODELS_MAPPING)
STABLE_LM_MODELS_FAMILY = {model: 'stable-lm' for model in STABLE_LM_MODELS_MAPPING.keys()}
STABLE_LM_MODELS_CONTEXT_SIZE = {model: 4096 for model in STABLE_LM_MODELS_MAPPING.keys()}
_register_model('STABLE_LM')


# Pretrained StarCoder models
STAR_CODER_MODELS_MAPPING = {
    'star-coder-base': 'bigcode/starcoderbase',
    'star-coder': 'bigcode/starcoder',
    'star-coder-plus': 'bigcode/starcoderplus',
}
STAR_CODER_MODELS_DTYPES = {model: torch.bfloat16 for model in STAR_CODER_MODELS_MAPPING.keys()}
STAR_CODER_MODELS_PARAMS = {model: 15.5 for model in STAR_CODER_MODELS_MAPPING.keys()}
STAR_CODER_MODELS_FAMILY = {model: 'star-coder' for model in STAR_CODER_MODELS_MAPPING.keys()}
STAR_CODER_MODELS_CONTEXT_SIZE = {model: 8192 for model in STAR_CODER_MODELS_MAPPING.keys()}
STAR_CODER_MODELS_ADDITIONAL_MODEL_KWARGS = {
    'star-coder-base': {'trust_remote_code': True},
}
_register_model('STAR_CODER')


# Pretrained Star-chat models
STAR_CHAT_MODELS_MAPPING = {
    'star-chat-alpha': 'HuggingFaceH4/starchat-alpha',
    'star-chat-beta': 'HuggingFaceH4/starchat-beta',
}
STAR_CHAT_MODELS_DTYPES = {
    'star-chat-alpha': torch.float16,
    'star-chat-beta': torch.bfloat16,
}
STAR_CHAT_MODELS_PARAMS = {model: 15.5 for model in STAR_CHAT_MODELS_MAPPING.keys()}
STAR_CHAT_MODELS_FAMILY = {model: 'star-chat' for model in STAR_CHAT_MODELS_MAPPING.keys()}
STAR_CHAT_MODELS_CONTEXT_SIZE = {model: 8192 for model in STAR_CHAT_MODELS_MAPPING.keys()}
_register_model('STAR_CHAT')


# Pretrained GPT-2 models
GPT2_MODELS_MAPPING = {
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
}
GPT2_MODELS_DTYPES = {model: torch.float32 for model in GPT2_MODELS_MAPPING.keys()}
GPT2_MODELS_PARAMS = {
    'gpt2-medium': 355/1e3,
    'gpt2-large': 774/1e3,
    'gpt2-xl': 1.5,
}
GPT2_MODELS_FAMILY = {model: 'gpt2' for model in GPT2_MODELS_MAPPING.keys()}
GPT2_MODELS_CONTEXT_SIZE = {model: 1024 for model in GPT2_MODELS_MAPPING.keys()}
_register_model('GPT2')


# Pretrained GPT-J and GPT-Neo models
GPT_J_AND_NEO_MODELS_MAPPING = {
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neo-125M': 'EleutherAI/gpt-neo-125m',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neoX-20B': 'EleutherAI/gpt-neox-20b',
}
GPT_J_AND_NEO_MODELS_DTYPES = {
    'gpt-j-6B': torch.float32,
    'gpt-neo-125M': torch.float32,
    'gpt-neo-1.3B': torch.float32,
    'gpt-neo-2.7B': torch.float32,
    'gpt-neoX-20B': torch.float16,
}
GPT_J_AND_NEO_MODELS_PARAMS = _infer_model_sizes(GPT_J_AND_NEO_MODELS_MAPPING)
GPT_J_AND_NEO_MODELS_FAMILY = {
    'gpt-j-6B': 'gpt-j',
    'gpt-neo-125M': 'gpt-neo',
    'gpt-neo-1.3B': 'gpt-neo',
    'gpt-neo-2.7B': 'gpt-neo',
    'gpt-neoX-20B': 'gpt-neo',
}
GPT_J_AND_NEO_MODELS_CONTEXT_SIZE = {model: 2048 for model in GPT_J_AND_NEO_MODELS_MAPPING.keys()}
_register_model('GPT_J_AND_NEO')


# Pretrained OPT models
OPT_MODELS_MAPPING = {
    'opt-125M': 'facebook/opt-125m',
    'opt-350M': 'facebook/opt-350m',
    'opt-1.3B': 'facebook/opt-1.3b',
    'opt-2.7B': 'facebook/opt-2.7b',
    'opt-6.7B': 'facebook/opt-6.7b',
    'opt-13B': 'facebook/opt-13b',
    'opt-30B': 'facebook/opt-30b',
    'opt-66B': 'facebook/opt-66b',
}
OPT_MODELS_DTYPES = {model: torch.float16 for model in OPT_MODELS_MAPPING.keys()}
OPT_MODELS_PARAMS = _infer_model_sizes(OPT_MODELS_MAPPING)
OPT_MODELS_FAMILY = {model: 'opt' for model in OPT_MODELS_MAPPING.keys()}
OPT_MODELS_CONTEXT_SIZE = {model: 2048 for model in OPT_MODELS_MAPPING.keys()}
_register_model('OPT')


# Pretrained CodeGEN models
CODEGEN_MODELS_MAPPING = {
    'codegen-350M': 'Salesforce/codegen-350M-mono',
    'codegen-2B': 'Salesforce/codegen-2B-mono',
    'codegen-6B': 'Salesforce/codegen-6B-mono',
    'codegen-16B': 'Salesforce/codegen-16B-mono',
}
CODEGEN_MODELS_DTYPES = {model: torch.float16 for model in CODEGEN_MODELS_MAPPING.keys()}
CODEGEN_MODELS_PARAMS = _infer_model_sizes(CODEGEN_MODELS_MAPPING)
CODEGEN_MODELS_FAMILY = {model: 'codegen' for model in CODEGEN_MODELS_MAPPING.keys()}
CODEGEN_MODELS_CONTEXT_SIZE = {model: 2048 for model in CODEGEN_MODELS_MAPPING.keys()}
_register_model('CODEGEN')


# Pretrained CodeGEN2 models
CODEGEN2_MODELS_MAPPING = {
    'codegen2-1B': 'Salesforce/codegen2-1B',
    'codegen2-3.7B': 'Salesforce/codegen2-3_7B',
    'codegen2-7B': 'Salesforce/codegen2-7B',
    'codegen2-16B': 'Salesforce/codegen2-16B',
    'codegen25-7B': 'Salesforce/codegen25-7B-mono',
    'codegen25-7B-instruct': 'Salesforce/codegen25-7b-instruct',
}
CODEGEN2_MODELS_DTYPES = {model: torch.float16 for model in CODEGEN2_MODELS_MAPPING.keys()}
CODEGEN2_MODELS_PARAMS = _infer_model_sizes(CODEGEN2_MODELS_MAPPING)
CODEGEN2_MODELS_FAMILY = {
    'codegen2-1B': 'codegen2',
    'codegen2-3.7B': 'codegen2',
    'codegen2-7B': 'codegen2',
    'codegen2-16B': 'codegen2',
    'codegen25-7B': 'codegen2.5',
    'codegen25-7B-instruct': 'codegen2.5',
}
CODEGEN2_MODELS_CONTEXT_SIZE = {model: 2048 for model in CODEGEN2_MODELS_MAPPING.keys()}
CODEGEN2_MODELS_VERSIONS = {
    'codegen25-7B': {'transformers': '<=4.33.3', 'tokenizers': '<=0.13.3'},
    'codegen25-7B-instruct': {'transformers': '<=4.33.3', 'tokenizers': '<=0.13.3'},
}
CODEGEN2_MODELS_ADDITIONAL_MODEL_KWARGS = {
    'codegen2-1B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-3.7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-16B': {'trust_remote_code': True, 'revision': 'main'},
}
CODEGEN2_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {
    'codegen25-7B': {'trust_remote_code': True},
    'codegen25-7B-instruct': {'trust_remote_code': True},
}
_register_model('CODEGEN2')


# Pretrained Vicuna (1.3) models
VICUNA_MODELS_MAPPING = {
    'vicuna-7B': 'lmsys/vicuna-7b-v1.3',
    'vicuna-13B': 'lmsys/vicuna-13b-v1.3',
}
VICUNA_MODELS_DTYPES = {model: torch.float16 for model in VICUNA_MODELS_MAPPING.keys()}
VICUNA_MODELS_PARAMS = _infer_model_sizes(VICUNA_MODELS_MAPPING)
VICUNA_MODELS_FAMILY = {model: 'vicuna1.3' for model in VICUNA_MODELS_MAPPING.keys()}
VICUNA_MODELS_CONTEXT_SIZE = {model: 2048 for model in VICUNA_MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
VICUNA_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in VICUNA_MODELS_MAPPING.keys()}
_register_model('VICUNA')


# Pretrained llama-2 models
LLAMA2_MODELS_MAPPING = {
    'llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'llama2-13B': 'meta-llama/Llama-2-13b-hf',
    'llama2-70B': 'meta-llama/Llama-2-70b-hf',
    'llama2-7B-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13B-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70B-chat': 'meta-llama/Llama-2-70b-chat-hf',
}
LLAMA2_MODELS_DTYPES = {model: torch.float16 for model in LLAMA2_MODELS_MAPPING.keys()}
LLAMA2_MODELS_PARAMS = _infer_model_sizes(LLAMA2_MODELS_MAPPING)
LLAMA2_MODELS_FAMILY = {model: 'llama2' for model in LLAMA2_MODELS_MAPPING.keys()}
LLAMA2_MODELS_CONTEXT_SIZE = {model: 4096 for model in LLAMA2_MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
LLAMA2_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in LLAMA2_MODELS_MAPPING.keys()}
_register_model('LLAMA2')


# Code-Llama models (based on llama2 models)
CODE_LLAMA_MODELS_MAPPING = {
    'code-llama-7B': 'codellama/CodeLlama-7b-hf',
    'code-llama-13B': 'codellama/CodeLlama-13b-hf',
    'code-llama-34B': 'codellama/CodeLlama-34b-hf',
    'code-llama-70B': 'codellama/CodeLlama-70b-hf',
    'code-llama-7B-python': 'codellama/CodeLlama-7b-Python-hf',
    'code-llama-13B-python': 'codellama/CodeLlama-13b-Python-hf',
    'code-llama-34B-python': 'codellama/CodeLlama-34b-Python-hf',
    'code-llama-70B-python': 'codellama/CodeLlama-70b-Python-hf',
    'code-llama-7B-instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'code-llama-13B-instruct': 'codellama/CodeLlama-13b-Instruct-hf',
    'code-llama-34B-instruct': 'codellama/CodeLlama-34b-Instruct-hf',
    'code-llama-70B-instruct': 'codellama/CodeLlama-70b-Instruct-hf',
}
CODE_LLAMA_MODELS_DTYPES = {model: torch.bfloat16 for model in CODE_LLAMA_MODELS_MAPPING.keys()}
CODE_LLAMA_MODELS_PARAMS = _infer_model_sizes(CODE_LLAMA_MODELS_MAPPING)
CODE_LLAMA_MODELS_FAMILY = {model: 'code-llama' for model in CODE_LLAMA_MODELS_MAPPING.keys()}
CODE_LLAMA_MODELS_CONTEXT_SIZE = {model: 4096 for model in CODE_LLAMA_MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
CODE_LLAMA_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in CODE_LLAMA_MODELS_MAPPING.keys()}
CODE_LLAMA_MODELS_VERSIONS = {
    'code-llama-70B': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
    'code-llama-70B-python': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
    'code-llama-70B-instruct': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
}
_register_model('CODE_LLAMA')


# Mistral models
MISTRAL_MODELS_MAPPING = {
    'mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'mistral-7B-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
}
MISTRAL_MODELS_DTYPES = {model: torch.bfloat16 for model in MISTRAL_MODELS_MAPPING.keys()}
MISTRAL_MODELS_PARAMS = _infer_model_sizes(MISTRAL_MODELS_MAPPING)
MISTRAL_MODELS_FAMILY = {model: 'mistral' for model in MISTRAL_MODELS_MAPPING.keys()}
MISTRAL_MODELS_CONTEXT_SIZE = {model: 8192 for model in MISTRAL_MODELS_MAPPING.keys()}
MISTRAL_MODELS_VERSIONS = {model: {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'} for model in MISTRAL_MODELS_MAPPING.keys()}
MISTRAL_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MISTRAL_MODELS_MAPPING.keys()}
_register_model('MISTRAL')


# Zephyr models
ZEPHYR_MODELS_MAPPING = {
    'zephyr-7B-alpha': 'HuggingFaceH4/zephyr-7b-alpha',
    'zephyr-7B-beta': 'HuggingFaceH4/zephyr-7b-beta',
}
ZEPHYR_MODELS_DTYPES = {model: torch.bfloat16 for model in ZEPHYR_MODELS_MAPPING.keys()}
ZEPHYR_MODELS_PARAMS = _infer_model_sizes(ZEPHYR_MODELS_MAPPING)
ZEPHYR_MODELS_FAMILY = {model: 'zephyr' for model in ZEPHYR_MODELS_MAPPING.keys()}
ZEPHYR_MODELS_CONTEXT_SIZE = {model: 8192 for model in ZEPHYR_MODELS_MAPPING.keys()}
ZEPHYR_MODELS_VERSIONS = {
    'zephyr-7B-alpha': {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'},
    'zephyr-7B-beta': {'transformers': '>=4.35.0', 'tokenizers': '>=0.14.0'},
}
ZEPHYR_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in ZEPHYR_MODELS_MAPPING.keys()}
_register_model('ZEPHYR')


# Summarize all supported model names
ALLOWED_MODELS = tuple(ALL_MODELS_MAPPING.keys())

ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)



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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(ALL_MODELS_MAPPING[model_name], device_map=device_map,
                                                 torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                 load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                 **additional_kwargs)
    
    # If the flag is active we directly put our model on one gpu without using any device_map (this is 
    # more efficient). But if the model is quantized, this is already done automatically because quantization
    # happen only on gpu
    if only_move_to_one_gpu and not quantization:
        # This operation is in-place for nn.Module
        model.cuda(gpu_rank)

    # For some reason bettertransformer is supported for codegen2 models but makes them crash during the forward
    if not ('codegen2-' in model_name):
        # Convert to better transformer to use Pytorch optimizations if supported by the model
        try:
            model = model.to_bettertransformer()
        except:
            pass
        
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


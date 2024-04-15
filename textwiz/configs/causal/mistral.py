import torch

from .. import _infer_model_sizes

# Mistral models
MODELS_MAPPING = {
    'mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'mistral-7B-instruct-v1': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral-7B-instruct-v2': 'mistralai/Mistral-7B-Instruct-v0.2',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'mistral' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {
    'mistral-7B': 8192,
    'mistral-7B-instruct-v1': 8192,
    'mistral-7B-instruct-v2': 32768,
}


MODELS_VERSIONS = {
    'mistral-7B': {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'},
    'mistral-7B-instruct-v1': {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'},
    'mistral-7B-instruct-v2': {'transformers': '>=4.36.0', 'tokenizers': '>=0.14.0'},
}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
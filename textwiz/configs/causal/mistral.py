import torch

from .. import _infer_model_sizes

# Mistral models
MODELS_MAPPING = {
    'mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'mistral-7B-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'mistral' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8192 for model in MODELS_MAPPING.keys()}


MODELS_VERSIONS = {model: {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'} for model in MODELS_MAPPING.keys()}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
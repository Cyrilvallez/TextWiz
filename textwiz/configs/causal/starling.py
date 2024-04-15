import torch

from .. import _infer_model_sizes

# Mistral models
MODELS_MAPPING = {
    'starling-7B-beta': 'Nexusflow/Starling-LM-7B-beta',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'starling' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8192 for model in MODELS_MAPPING.keys()}


MODELS_VERSIONS = {
    'starling-7B-beta': {'transformers': '>=4.37.1', 'tokenizers': '>=0.14.0'},
}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
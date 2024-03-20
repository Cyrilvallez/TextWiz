import torch

from .. import _infer_model_sizes


# Zephyr models
MODELS_MAPPING = {
    'zephyr-7B-alpha': 'HuggingFaceH4/zephyr-7b-alpha',
    'zephyr-7B-beta': 'HuggingFaceH4/zephyr-7b-beta',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'zephyr' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8192 for model in MODELS_MAPPING.keys()}
MODELS_VERSIONS = {
    'zephyr-7B-alpha': {'transformers': '>=4.34.0', 'tokenizers': '>=0.14.0'},
    'zephyr-7B-beta': {'transformers': '>=4.35.0', 'tokenizers': '>=0.14.0'},
}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
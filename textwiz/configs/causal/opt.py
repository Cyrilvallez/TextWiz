import torch

from .. import _infer_model_sizes

# Pretrained OPT models
MODELS_MAPPING = {
    'opt-125M': 'facebook/opt-125m',
    'opt-350M': 'facebook/opt-350m',
    'opt-1.3B': 'facebook/opt-1.3b',
    'opt-2.7B': 'facebook/opt-2.7b',
    'opt-6.7B': 'facebook/opt-6.7b',
    'opt-13B': 'facebook/opt-13b',
    'opt-30B': 'facebook/opt-30b',
    'opt-66B': 'facebook/opt-66b',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'opt' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
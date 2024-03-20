import torch

from .. import _infer_model_sizes

# Pretrained StableLM models
MODELS_MAPPING = {
    'stable-lm-3B': 'stabilityai/stablelm-base-alpha-3b',
    'stable-lm-7B': 'stabilityai/stablelm-base-alpha-7b',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'stable-lm' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 4096 for model in MODELS_MAPPING.keys()}
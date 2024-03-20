import torch

from .. import _infer_model_sizes

# Pretrained bloom models
MODELS_MAPPING = {
    'bloom-560M': 'bigscience/bloom-560m',
    'bloom-1.7B': 'bigscience/bloom-1b7',
    'bloom-3B': 'bigscience/bloom-3b',
    'bloom-7.1B':'bigscience/bloom-7b1',
    'bloom-176B': 'bigscience/bloom',
}
MODELS_DTYPES = {
    'bloom-560M': torch.float16,
    'bloom-1.7B': torch.float16,
    'bloom-3B': torch.float16,
    'bloom-7.1B':torch.float16,
    'bloom-176B': torch.bfloat16,
}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'bloom' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
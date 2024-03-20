import torch

from .. import _infer_model_sizes

# Pretrained GPT-J and GPT-Neo models
MODELS_MAPPING = {
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neo-125M': 'EleutherAI/gpt-neo-125m',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neoX-20B': 'EleutherAI/gpt-neox-20b',
}
MODELS_DTYPES = {
    'gpt-j-6B': torch.float32,
    'gpt-neo-125M': torch.float32,
    'gpt-neo-1.3B': torch.float32,
    'gpt-neo-2.7B': torch.float32,
    'gpt-neoX-20B': torch.float16,
}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {
    'gpt-j-6B': 'gpt-j',
    'gpt-neo-125M': 'gpt-neo',
    'gpt-neo-1.3B': 'gpt-neo',
    'gpt-neo-2.7B': 'gpt-neo',
    'gpt-neoX-20B': 'gpt-neo',
}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
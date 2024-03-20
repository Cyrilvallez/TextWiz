import torch

# Pretrained GPT-2 models
MODELS_MAPPING = {
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
}
MODELS_DTYPES = {model: torch.float32 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = {
    'gpt2-medium': 355/1e3,
    'gpt2-large': 774/1e3,
    'gpt2-xl': 1.5,
}
MODELS_FAMILY = {model: 'gpt2' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 1024 for model in MODELS_MAPPING.keys()}
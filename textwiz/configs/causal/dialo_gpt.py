import torch

# Pretrained Dialo-GPT models
MODELS_MAPPING = {
    'dialo-gpt-small': 'microsoft/DialoGPT-small',
    'dialo-gpt-medium': 'microsoft/DialoGPT-medium',
    'dialo-gpt-large': 'microsoft/DialoGPT-large',
}
MODELS_DTYPES = {model: torch.float32 for model in MODELS_MAPPING}
MODELS_PARAMS = {
    'dialo-gpt-small': 125/1e3,
    'dialo-gpt-medium': 355/1e3,
    'dialo-gpt-large': 775/1e3,
}
MODELS_FAMILY = {model: 'dialo-gpt' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 1024 for model in MODELS_MAPPING.keys()}
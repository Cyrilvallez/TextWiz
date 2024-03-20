import torch

# Pretrained StarCoder models
MODELS_MAPPING = {
    'star-coder-base': 'bigcode/starcoderbase',
    'star-coder': 'bigcode/starcoder',
    'star-coder-plus': 'bigcode/starcoderplus',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = {model: 15.5 for model in MODELS_MAPPING.keys()}
MODELS_FAMILY = {model: 'star-coder' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 8192 for model in MODELS_MAPPING.keys()}
MODELS_ADDITIONAL_MODEL_KWARGS = {
    'star-coder-base': {'trust_remote_code': True},
}
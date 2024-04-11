import torch

from .. import _infer_model_sizes


# Cohere models
MODELS_MAPPING = {
    'command-r': 'CohereForAI/c4ai-command-r-v01',
    'command-r-plus': 'CohereForAI/c4ai-command-r-plus',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = {
    'command-r': 35,
    'command-r-plus': 104,
}
MODELS_FAMILY = {model: 'command' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 131072 for model in MODELS_MAPPING.keys()}
MODELS_VERSIONS = {
    'command-r-plus': {'transformers': '>=4.40', 'tokenizers': '>=0.15.0'},
}
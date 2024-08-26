import torch

from .. import _infer_model_sizes

# Pretrained llama-3.1 models
MODELS_MAPPING = {
    'llama3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'llama3.1-70B': 'meta-llama/Meta-Llama-3.1-70B',
    'llama3.1-8B-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.1-70B-instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'llama3.1' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 131072 for model in MODELS_MAPPING.keys()}

MODELS_VERSIONS = {model: {'transformers': '>=4.43.0', 'tokenizers': '>=0.19.1'} for model in MODELS_MAPPING.keys()}

# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
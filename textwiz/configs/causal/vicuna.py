import torch

from .. import _infer_model_sizes

# Pretrained Vicuna (1.3) models
MODELS_MAPPING = {
    'vicuna-7B': 'lmsys/vicuna-7b-v1.3',
    'vicuna-13B': 'lmsys/vicuna-13b-v1.3',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'vicuna1.3' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
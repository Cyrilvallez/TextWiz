import torch

from .. import _infer_model_sizes

# Pretrained llama-2 models
MODELS_MAPPING = {
    'llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'llama2-13B': 'meta-llama/Llama-2-13b-hf',
    'llama2-70B': 'meta-llama/Llama-2-70b-hf',
    'llama2-7B-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13B-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70B-chat': 'meta-llama/Llama-2-70b-chat-hf',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'llama2' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 4096 for model in MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
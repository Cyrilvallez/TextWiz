import torch

from .. import _infer_model_sizes

# Code-Llama models (based on llama2 models)
MODELS_MAPPING = {
    'code-llama-7B': 'codellama/CodeLlama-7b-hf',
    'code-llama-13B': 'codellama/CodeLlama-13b-hf',
    'code-llama-34B': 'codellama/CodeLlama-34b-hf',
    'code-llama-70B': 'codellama/CodeLlama-70b-hf',
    'code-llama-7B-python': 'codellama/CodeLlama-7b-Python-hf',
    'code-llama-13B-python': 'codellama/CodeLlama-13b-Python-hf',
    'code-llama-34B-python': 'codellama/CodeLlama-34b-Python-hf',
    'code-llama-70B-python': 'codellama/CodeLlama-70b-Python-hf',
    'code-llama-7B-instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'code-llama-13B-instruct': 'codellama/CodeLlama-13b-Instruct-hf',
    'code-llama-34B-instruct': 'codellama/CodeLlama-34b-Instruct-hf',
    'code-llama-70B-instruct': 'codellama/CodeLlama-70b-Instruct-hf',
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'code-llama' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 4096 for model in MODELS_MAPPING.keys()}
# Fast llama tokenizers are buggy in current transformers versions
# TODO: may need to be changed in future versions if they correct the bug
ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
MODELS_VERSIONS = {
    'code-llama-70B': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
    'code-llama-70B-python': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
    'code-llama-70B-instruct': {'transformers': '>=4.37.1', 'tokenizers': '>=0.15.1'},
}
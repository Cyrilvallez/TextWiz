import torch

from .. import _infer_model_sizes


MODELS_MAPPING = {
    'SFR-Embedding-Mistral': 'Salesforce/SFR-Embedding-Mistral'
}
MODELS_DTYPES = {model: torch.bfloat16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = {model: 7 for model in MODELS_MAPPING.keys()}
MODELS_FAMILY = {model: 'mistral' for model in MODELS_MAPPING.keys()}

# We want to stay inside the context sliding window for maximum embedding accuracy
MODELS_CONTEXT_SIZE = {model: 4096 for model in MODELS_MAPPING.keys()}


MODELS_VERSIONS = {model: {'transformers': '>=4.37.0', 'tokenizers': '>=0.14.0'} for model in MODELS_MAPPING.keys()}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in MODELS_MAPPING.keys()}
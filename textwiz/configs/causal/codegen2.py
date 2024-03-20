import torch

from .. import _infer_model_sizes

# Pretrained CodeGEN2 models
MODELS_MAPPING = {
    'codegen2-1B': 'Salesforce/codegen2-1B',
    'codegen2-3.7B': 'Salesforce/codegen2-3_7B',
    'codegen2-7B': 'Salesforce/codegen2-7B',
    'codegen2-16B': 'Salesforce/codegen2-16B',
    'codegen25-7B': 'Salesforce/codegen25-7B-mono',
    'codegen25-7B-instruct': 'Salesforce/codegen25-7b-instruct',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {
    'codegen2-1B': 'codegen2',
    'codegen2-3.7B': 'codegen2',
    'codegen2-7B': 'codegen2',
    'codegen2-16B': 'codegen2',
    'codegen25-7B': 'codegen2.5',
    'codegen25-7B-instruct': 'codegen2.5',
}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
MODELS_VERSIONS = {
    'codegen25-7B': {'transformers': '<=4.33.3', 'tokenizers': '<=0.13.3'},
    'codegen25-7B-instruct': {'transformers': '<=4.33.3', 'tokenizers': '<=0.13.3'},
}
MODELS_ADDITIONAL_MODEL_KWARGS = {
    'codegen2-1B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-3.7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-7B': {'trust_remote_code': True, 'revision': 'main'},
    'codegen2-16B': {'trust_remote_code': True, 'revision': 'main'},
}
MODELS_ADDITIONAL_TOKENIZER_KWARGS = {
    'codegen25-7B': {'trust_remote_code': True},
    'codegen25-7B-instruct': {'trust_remote_code': True},
}
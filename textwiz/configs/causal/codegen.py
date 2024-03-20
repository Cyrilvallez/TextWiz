import torch

from .. import _infer_model_sizes

# Pretrained CodeGEN models
MODELS_MAPPING = {
    'codegen-350M': 'Salesforce/codegen-350M-mono',
    'codegen-2B': 'Salesforce/codegen-2B-mono',
    'codegen-6B': 'Salesforce/codegen-6B-mono',
    'codegen-16B': 'Salesforce/codegen-16B-mono',
}
MODELS_DTYPES = {model: torch.float16 for model in MODELS_MAPPING.keys()}
MODELS_PARAMS = _infer_model_sizes(MODELS_MAPPING)
MODELS_FAMILY = {model: 'codegen' for model in MODELS_MAPPING.keys()}
MODELS_CONTEXT_SIZE = {model: 2048 for model in MODELS_MAPPING.keys()}
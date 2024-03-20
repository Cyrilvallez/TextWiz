import re
import importlib


def _infer_model_size(model_name: str) -> float:
    """Return the number of parameters a model has from its name if it can be inferred from it. Raise a 
    ValueError otherwise.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    float
        The number of parameters of the model, in billions.
    """

    # The following regex matches any digits possibly separated with a dot ('.') which is immeditely
    # followed by a 'B' or 'M' to capture the model size following our model name convention. Parenthesis 
    # allow to capture given groups of the regex thanks to the match object .group() method.
    pattern = r'([0-9]+(?:\.[0-9]+)?)([BM])'

    match = re.search(pattern, model_name)
    if match:
        matched_number = match.group(1)
        matched_letter = match.group(2)
        # Model size in billion (B) of parameters
        model_size = float(matched_number) if matched_letter == 'B' else float(matched_number)/1e3
        return model_size
    else:
        raise ValueError('The model number of parameters cannot be inferred from its name.')
    

def _infer_model_sizes(name_mapping: dict[str, str]) -> dict[str, float]:
    """Infer the number of parameters of all model names (dict keys) and return them as {key: #params}.

    Parameters
    ----------
    name_mapping : dict[str, str]
        A dictionary whose keys are the model names.

    Returns
    -------
    dict[str, float]
        A mapping from names to number of parameters.
    """

    return {key: _infer_model_size(key) for key in name_mapping.keys()}


def _register_config(config_name: str):
    """_summary_

    Parameters
    ----------
    config_name : str
        _description_
    """

    mod = importlib.import_module('.' + config_name, package=__package__)

    required_params = ('ALL_MODELS_MAPPING', 'ALL_MODELS_DTYPES', 'ALL_MODELS_PARAMS', 'ALL_MODELS_FAMILY',
                       'ALL_MODELS_CONTEXT_SIZE', 'ALL_MODELS_VERSIONS', 'ALL_MODELS_ADDITIONAL_MODEL_KWARGS',
                       'ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS', 'ALL_MODELS_PURPOSE')

    # Template string to `exec` in order to merge the dictionaries
    template = '{param}.update(mod.{param})'

    for param in required_params:
        code = template.format(param=param)
        try:
            exec(code)
        except:
            raise ValueError(f'Missing entry "{param}" in config file "{config_name}"')


# Those will be updated with each call to _register_config()
ALL_MODELS_MAPPING = {}
ALL_MODELS_DTYPES = {}
ALL_MODELS_PARAMS = {}
ALL_MODELS_FAMILY = {}
ALL_MODELS_CONTEXT_SIZE = {}
ALL_MODELS_VERSIONS = {}
ALL_MODELS_ADDITIONAL_MODEL_KWARGS = {}
ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {}
ALL_MODELS_PURPOSE = {}

# Only those variables are imported with wildcard
__all__ = [
    ALL_MODELS_MAPPING,
    ALL_MODELS_DTYPES,
    ALL_MODELS_PARAMS,
    ALL_MODELS_FAMILY,
    ALL_MODELS_CONTEXT_SIZE,
    ALL_MODELS_VERSIONS,
    ALL_MODELS_ADDITIONAL_MODEL_KWARGS,
    ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS,
    ALL_MODELS_PURPOSE,
]

# Register the configs
_register_config('causal')
_register_config('embedding')
import importlib
import os

def _register_model(filename: str):
    """Register a model family into the global variables containing all models parameters.

    Parameters
    ----------
    model_name : str
        Name of the file containing the model family parameters.
    """

    mod = importlib.import_module('.' + filename, package=__package__)

    required_params = ('MODELS_MAPPING', 'MODELS_DTYPES', 'MODELS_PARAMS', 'MODELS_FAMILY', 'MODELS_CONTEXT_SIZE')
    optional_params = ('MODELS_VERSIONS', 'MODELS_ADDITIONAL_MODEL_KWARGS', 'MODELS_ADDITIONAL_TOKENIZER_KWARGS')

    # Template string to `exec` in order to merge the dictionaries
    template = 'ALL_{param}.update(mod.{param})'

    for param in required_params:
        code = template.format(param=param)
        try:
            exec(code)
        except:
            raise ValueError(f'Missing entry "{param}" in config file "{filename}"')

    for param in optional_params:
        try:
            code = template.format(param=param)
            exec(code)
        # In this case the optional variable is not present
        except AttributeError:
            pass


# Those will be updated with each call to _register_model()
ALL_MODELS_MAPPING = {}
ALL_MODELS_DTYPES = {}
ALL_MODELS_PARAMS = {}
ALL_MODELS_FAMILY = {}
ALL_MODELS_CONTEXT_SIZE = {}
ALL_MODELS_VERSIONS = {}
ALL_MODELS_ADDITIONAL_MODEL_KWARGS = {}
ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {}


# Register all models contained in the folder
for config_file in os.listdir(os.path.dirname(__file__)):
    if config_file.endswith('.py') and config_file != '__init__.py':
        _register_model(config_file.rsplit('.py', 1)[0])


ALL_MODELS_PURPOSE = {model: 'causal' for model in ALL_MODELS_MAPPING.keys()}
from textgen.generation import HFModel
from textgen import loader
from textgen import prompt_template
# import it here so that the warnings are suppressed when doing `import engine`
from textgen import warnings_suppressor


def estimate_number_of_gpus(models: list[str], quantization_8bits: bool = False, quantization_4bits: bool = False,
                            max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8) -> list[int]:
    """Estimate the mumber of gpus needed to run each of the `models` correctly.

    Parameters
    ----------
    models : list[str]
        The models.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    list[int]
        The number of gpus for each model.
    """
    
    model_footprints = []
    for model in models:
        # Override quantization for bloom because it's too big to load in float16
        if model == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=True, quantization_4bits=False,
                                                                max_fraction_gpu_0=0.9,
                                                                max_fraction_gpus=0.9)
        else:
            gpu_needed, _ = loader.estimate_model_gpu_footprint(model, quantization_8bits=quantization_8bits,
                                                                quantization_4bits=quantization_4bits,
                                                                max_fraction_gpu_0=max_fraction_gpu_0,
                                                                max_fraction_gpus=max_fraction_gpus)
        model_footprints.append(gpu_needed)

    return model_footprints



def is_chat_model(model_name: str) -> bool:
    """Check if given model is chat-optimized.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
    bool
        Whether the model is chat optimized or not.
    """

    if model_name == 'codegen25-7B-instruct':
        return True

    template = prompt_template.get_prompt_template(model_name)
    return template.default_mode == 'chat'



# Relatively small models (they should fit on a single A100 GPU)
SMALL_MODELS = tuple(model for model in loader.ALLOWED_MODELS if loader.ALL_MODELS_PARAMS[model] <= 16)
# Large models (they require more than 1 A100 GPU)
LARGE_MODELS = tuple(model for model in loader.ALLOWED_MODELS if loader.ALL_MODELS_PARAMS[model] > 16)

assert set(loader.ALLOWED_MODELS) == set(SMALL_MODELS + LARGE_MODELS), 'We are somehow missing some models...'


# Model with non-default prompt template
SMALL_MODELS_SPECIAL_PROMPT = tuple(model for model in SMALL_MODELS if model in prompt_template.PROMPT_MAPPING.keys())
LARGE_MODELS_SPECIAL_PROMPT = tuple(model for model in LARGE_MODELS if model in prompt_template.PROMPT_MAPPING.keys())



# Models that we decided to keep for further code benchmarks
GOOD_CODERS = (
    'star-coder-base',
    'star-coder',
    'star-chat-alpha',
    'codegen-16B',
    'codegen25-7B',
    'codegen25-7B-instruct',
    'code-llama-34B',
    'code-llama-34B-python',
    'code-llama-34B-instruct',
    'llama2-70B',
    'llama2-70B-chat',
)


SMALL_GOOD_CODERS = tuple(model for model in GOOD_CODERS if model in SMALL_MODELS)
LARGE_GOOD_CODERS = tuple(model for model in GOOD_CODERS if model in LARGE_MODELS)


assert set(GOOD_CODERS) == set(SMALL_GOOD_CODERS + LARGE_GOOD_CODERS), 'We are somehow missing some good coder models...'


# Model that we decided to keep for further code benchmarks with non-default prompt template
SMALL_GOOD_CODERS_SPECIAL_PROMPT = tuple(model for model in SMALL_GOOD_CODERS if model in prompt_template.PROMPT_MAPPING.keys())
LARGE_GOOD_CODERS_SPECIAL_PROMPT = tuple(model for model in LARGE_GOOD_CODERS if model in prompt_template.PROMPT_MAPPING.keys())
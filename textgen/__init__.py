# Define easy entry points for most useful features
from textgen.generation import HFModel
from textgen.loader import load_model, load_tokenizer, load_model_and_tokenizer, estimate_model_gpu_footprint
from textgen.conversation_template import get_empty_conversation_template, get_conversation_from_yaml_template
from textgen.prompt_template import get_prompt_template
from textgen.stopping import StoppingType, create_stopping_criteria, post_process_sequences
from textgen.code_parser import PythonParser
from textgen.streamer import TextContinuationStreamer
# import it here so that the warnings are suppressed when doing `import engine`
from textgen import warnings_suppressor


__version__ = '0.0.1'


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

    template = get_prompt_template(model_name)
    return template.default_mode == 'chat'
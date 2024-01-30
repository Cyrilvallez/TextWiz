"""
This file contains the prompt templates for the models we use, for causal generation. These
templates are especially not meant for conversations with the models, only for prompt completion, without
memory of previous prompts.
"""

from .loader import ALLOWED_MODELS

PROMPT_MODES = ('default', 'generation', 'infill', 'chat')


class GenericPromptTemplate(object):

    def __init__(self, mode: str = 'default'):

        if mode not in PROMPT_MODES:
            raise ValueError(f'The mode for creating the prompt must be one of {*PROMPT_MODES,}')
        
        self.mode = mode
        self.default_mode = 'generation'
        self.extra_eos_tokens = []


    def get_prompt(self, prompt: str, model_context: str = '', suffix: str = '', system_prompt: str = '') -> str:
        """Format the `prompt` according to `self.mode`.

        Parameters
        ----------
        prompt : str
            The prompt to format.
        model_context : str, optional
            An optional context forming the start of the model answer. By default ''.
        suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all modes except `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all modes
            except `chat`, by default ''.

        Returns
        -------
        str
            Formatted prompt.
        """

        if self.mode == 'default':
            return self.format_default(prompt, model_context=model_context, suffix=suffix, system_prompt=system_prompt)
        elif self.mode == 'generation':
            return self.format_generation(prompt, model_context=model_context)
        elif self.mode == 'infill':
            return self.format_infill(prompt, model_context=model_context, suffix=suffix)
        elif self.mode == 'chat':
            return self.format_chat(prompt, model_context=model_context, system_prompt=system_prompt)
        

    def format_default(self, prompt: str, model_context: str = '', suffix: str = '', system_prompt: str = '') -> str:
        """Format the `prompt` when `self.mode = 'default'`.

        Parameters
        ----------
        prompt : str
            The prompt to format.
        model_context : str, optional
            An optional context forming the start of the model answer. By default ''.
        suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all modes except `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all modes
            except `chat`, by default ''.

        Returns
        -------
        str
            Formatted prompt.
        """

        if self.default_mode == 'generation':
            return self.format_generation(prompt, model_context=model_context)
        elif self.default_mode == 'infill':
            return self.format_infill(prompt, model_context=model_context, suffix=suffix)
        elif self.default_mode == 'chat':
            return self.format_chat(prompt, model_context=model_context, system_prompt=system_prompt)


    def format_generation(self, prompt: str, model_context: str = '') -> str:
        """Format the prompt for usual models.

        Parameters
        ----------
        prompt : str
            Prompt to format
        model_context : str
            An optional context. This is simply appended to `prompt`. By default ''.

        Returns
        -------
        str
            Formatted prompt.
        """
        return prompt + model_context
    

    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:
        """Format the prompt for chat models.

        Parameters
        ----------
        prompt : str
            The prompt to format.
        model_context : str, optional
            An optional context forming the start of the model answer, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning, by default ''.

        Returns
        -------
        str
            Formatted prompt.
        """
        raise RuntimeError(f'Chat mode not supported for {self.__class__.__name__}.')
    

    def format_infill(self, prefix: str, model_context: str = '', suffix: str = '') -> str:
        """Format the prompt for models supporting infilling.

        Parameters
        ----------
        prompt : str
            The prompt to format.
        suffix : str, optional
            An optional suffix to form the prompt, by default ''.
        model_context : str, optional
            An optional context forming the start of the model answer, by default ''.
        
        Returns
        -------
        str
            Formatted prompt.
        """
        raise RuntimeError(f'Infill mode not supported for {self.__class__.__name__}.')
    

    def get_extra_eos(self) -> list[str]:
        """Return the potential extra eos tokens upon which to stop generation.
        """
        return self.extra_eos_tokens
    

    def set_mode(self, mode: str):
        """Set the formatting mode.
        """
        if mode not in PROMPT_MODES:
            raise ValueError(f'The mode for creating the prompt must be one of {*PROMPT_MODES,}')
        self.mode = mode
    

# DialoGPT template: https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Mariama%21+How+are+you%3F
class DialoGPTPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.eos_token = '<|endoftext|>'

    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:
        # The system prompt is ignored, and the model context is just appended to the prompt
        # This is because DialoGPT is a an extremely basic chat model which does not really handle these parameters
        return prompt + model_context + self.eos_token
    

# StarCoder template: https://huggingface.co/bigcode/starcoder
class StarCoderPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        # self.default_mode = 'infill'
        self.default_mode = 'generation'

        self.prefix_token = '<fim_prefix>'
        self.suffix_token = '<fim_suffix>'
        self.middle_token = '<fim_middle>'

    def format_infill(self, prefix: str, model_context: str = '', suffix: str = '') -> str:

        return self.prefix_token + prefix + self.suffix_token + suffix + self.middle_token + model_context
    
    

# Starchat prompt modeling (see https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py)
# See also FastChat (/https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#817) but note that
# it was modified by me (https://github.com/lm-sys/FastChat/pull/2239)
class StarChatPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_token = '<|system|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.sep_token = '<|end|>'
        self.extra_eos_tokens = [self.sep_token]

    
    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        system = self.system_token + '\n' + system_prompt + self.sep_token + '\n' if system_prompt != '' else ''

        return system + self.user_token + '\n' + prompt + self.sep_token + '\n' + self.assistant_token + '\n' + model_context
    

# Codegen2 template: https://huggingface.co/Salesforce/codegen2-3_7B
class Codegen2PromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        # Keep the default to generation as the infill mode seems to be worse (at least on HumanEval)
        # self.default_mode = 'infill'
        self.default_mode = 'generation'

        self.mask_token = '<mask_1>'
        self.eos_token = '<|endoftext|>'
        self.sep_token = '<sep>'
        self.extra_eos_tokens = ['<eom>']

    def format_infill(self, prefix: str, model_context: str = '', suffix: str = '') -> str:

        return prefix + self.mask_token + suffix + self.eos_token + self.sep_token + self.mask_token + model_context
    


# Vicuna 1.3 prompt modeling (https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py)
class VicunaPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.user_token = 'USER'
        self.assistant_token = 'ASSISTANT'

    
    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        if system_prompt == '':
            formatted_prompt = self.user_token + ': ' + prompt + ' ' + self.assistant_token + ':'
        else:
            formatted_prompt = system_prompt + ' ' + self.user_token + ': ' + prompt + ' ' + self.assistant_token + ':'

        if model_context != '':
            formatted_prompt += ' ' + model_context

        return formatted_prompt



# Llama2-chat prompt modeling (https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
# See also FastChat (https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L123) but note that
# it was modified by me (https://github.com/lm-sys/FastChat/pull/2239)
class Llama2ChatPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'

    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        if system_prompt.strip() != '':
            system_prompt = self.system_template.format(system_prompt=system_prompt.strip())
        else:
            system_prompt = ''

        formatted_prompt = self.user_token + ' ' + system_prompt + prompt.strip() + ' ' + self.assistant_token
        
        if model_context != '':
            formatted_prompt += ' ' + model_context

        return formatted_prompt
    


# references: https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf#chat_prompt
# and https://github.com/facebookresearch/codellama/blob/main/llama/generation.py#L506-L548
class CodeLlama70BInstructPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.system_token = 'Source: system'
        self.user_token = 'Source: user'
        self.assistant_token = 'Source: assistant'
        self.separator = '<step>'


    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        # If we are not using system prompt, the original code still add an empty string
        formatted_prompt = self.system_token + '\n\n ' + system_prompt.strip() + ' ' + self.separator + ' '

        formatted_prompt += self.user_token + '\n\n ' + prompt.strip() + ' ' + self.separator + ' '
        formatted_prompt += self.assistant_token + '\nDestination: user\n\n ' 
        
        if model_context != '':
            formatted_prompt += model_context

        return formatted_prompt



# reference: https://docs.mistral.ai/usage/guardrailing/
class MistralPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'


    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        system_prompt = system_prompt.strip()

        formatted_prompt = self.user_token + ' ' + system_prompt + ' ' + prompt.strip() + ' ' + self.assistant_token

        if model_context != '':
            prompt += ' ' + model_context

        return formatted_prompt


# reference: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
class ZephyrPromptTemplate(GenericPromptTemplate):

    def __init__(self, mode: str = 'default'):

        super().__init__(mode)
        self.default_mode = 'chat'

        self.eos_token = '</s>'
        self.system_token = '<|system|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'


    def format_chat(self, prompt: str, model_context: str = '', system_prompt: str = '') -> str:

        # If we are not using system prompt, do not add the template formatting with empty prompt

        if system_prompt.strip() != '':
            formatted_prompt = self.system_token + '\n' + system_prompt.strip() + self.eos_token + '\n'
        else:
            formatted_prompt = ''

        formatted_prompt += self.user_token + '\n' + prompt.strip() + self.eos_token + '\n' + self.assistant_token + '\n'

        if model_context != '':
            formatted_prompt += model_context

        return formatted_prompt

    

# Mapping from model name to prompt class name
PROMPT_MAPPING = {
    # DialoGPT
    'dialo-gpt-small': DialoGPTPromptTemplate,
    'dialo-gpt-medium': DialoGPTPromptTemplate,
    'dialo-gpt-large': DialoGPTPromptTemplate,

    # StarCoder
    'star-coder-base': StarCoderPromptTemplate,
    'star-coder': StarCoderPromptTemplate,
    'star-coder-plus': StarCoderPromptTemplate,

    # StarChat
    'star-chat-alpha': StarChatPromptTemplate,
    'star-chat-beta': StarChatPromptTemplate,

    # Codegen2
    'codegen2-1B': Codegen2PromptTemplate,
    'codegen2-3.7B': Codegen2PromptTemplate,
    'codegen2-7B': Codegen2PromptTemplate,
    'codegen2-16B': Codegen2PromptTemplate,
    'codegen25-7B': Codegen2PromptTemplate,
    'codegen25-7B-instruct': Codegen2PromptTemplate,

    # Vicuna (1.3)
    'vicuna-7B': VicunaPromptTemplate,
    'vicuna-13B': VicunaPromptTemplate,

    # Llama2-chat
    'llama2-7B-chat': Llama2ChatPromptTemplate,
    'llama2-13B-chat': Llama2ChatPromptTemplate,
    'llama2-70B-chat': Llama2ChatPromptTemplate,

    # Code-llama-instruct
    'code-llama-7B-instruct': Llama2ChatPromptTemplate,
    'code-llama-13B-instruct': Llama2ChatPromptTemplate,
    'code-llama-34B-instruct': Llama2ChatPromptTemplate,
    # Special template for the 70B version
    'code-llama-70B-instruct': CodeLlama70BInstructPromptTemplate,

    # Mistral
    'mistral-7B-instruct': MistralPromptTemplate,

    # Zephyr
    'zephyr-7B-alpha': ZephyrPromptTemplate,
    'zephyr-7B-beta': ZephyrPromptTemplate,
}


def get_prompt_template(model_name: str, mode: str = 'default') -> GenericPromptTemplate:
    """Return the prompt template class formating corresponding to `model_name`.

    Parameters
    ----------
    model_name : str
        Name of the current model.
    mode : str, optional
        The generation mode for the model, by default 'default'. Note that changing this value may cause
        issues as not all prompt templates support all modes.

    Returns
    -------
    GenericPromptTemplate
        A prompt template class corresponding to `model_name`.

    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    if mode not in PROMPT_MODES:
        raise ValueError(f'The mode for creating the prompt must be one of {*PROMPT_MODES,}')

    if model_name in PROMPT_MAPPING.keys():
        prompt = PROMPT_MAPPING[model_name](mode=mode)
    else:
        prompt = GenericPromptTemplate(mode)

    return prompt
"""
This file contains the conversation templates for the models we use.
"""
import uuid
import copy

from . import loader
from . import utils


class GenericConversation(object):
    """Class used to store a conversation with a model."""

    def __init__(self, eos_token: str):

        # Conversation history
        self.user_history_text = []
        self.model_history_text = []

        # Initial few-shot examples
        self.user_few_shot = []
        self.model_few_shot = []

        # system prompt
        self.system_prompt = ''

        # eos token
        self.eos_token = eos_token

        # Extra eos tokens
        self.extra_eos_tokens = []

        # Some templates need an additional space when using `get_last_turn_continuation_prompt`
        self.add_space_to_continuation_prompt = False

        # create unique identifier (used in gradio flagging)
        # TODO: maybe override __deepcopy__ method so that the id is different for deepcopies? For now it is
        # not needed but keep it in mind
        self.id = str(uuid.uuid4())


    @classmethod
    def from_yaml_template(cls, path: str, eos_token: str):
        """Set the system prompt and few-shot examples of a conversation from a yaml file.

        Parameters
        ----------
        path : str
            Path to the yaml file.
        eos_token : str
            EOS token for the current model.

        Returns
        -------
        GenericConversation
            The conversation with attributes set from the file.
        """

        conv = cls(eos_token)
        data = utils.load_yaml(path)

        if 'system_prompt' in data.keys():
            conv.set_system_prompt(data['system_prompt'])
        
        if 'few_shot_examples' in data.keys():
            few_shot_examples = data['few_shot_examples']
            user_few_shot = ['']*len(few_shot_examples)
            model_few_shot = ['']*len(few_shot_examples)

            for turn in few_shot_examples:
                if sorted(list(turn.keys())) != ['index', 'model', 'user']:
                    raise RuntimeError('The format of the yaml file is incorrect.')
                k = turn['index']
                user_few_shot[k] = turn['user']
                model_few_shot[k] = turn['model']

            conv.set_few_shot_examples(user_few_shot, model_few_shot)

        return conv


    def __len__(self) -> int:
        """Return the length of the current conversation. This does NOT include the few-shot examples.
        """
        return len(self.user_history_text)
    
    
    def __iter__(self):
        """Create a generator over (user_input, model_answer) tuples for all turns in the conversation,
        including the few-shot examples.
        """
        # Generate over copies so that the object in the class cannot change during iteration
        total_user_input = self.user_few_shot.copy() + self.user_history_text.copy()
        total_model_answer = self.model_few_shot.copy() + self.model_history_text.copy()

        for user_history, model_history in zip(total_user_input, total_model_answer):
            yield user_history, model_history

    
    def iter_without_few_shot(self):
        """Create a generator over (user_input, model_answer) tuples for all turns in the conversation,
        NOT including the few-shot examples.
        """
        for user_history, model_history in zip(self.user_history_text.copy(), self.model_history_text.copy()):
            yield user_history, model_history


    def __str__(self) -> str:
        """Format the conversation as a string.
        """

        N = len(self)

        if N == 0:
            return "The conversation is empty."
        
        else:
            out = ''
            for i, (user, model) in enumerate(self.iter_without_few_shot()):
                out += f'>> User: {user}\n'
                if model is not None:
                    out += f'>> Model: {model}'
                # Skip 2 lines between turns
                if i < N - 1:
                    out += '\n\n'

            return out
        

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        

    def append_user_message(self, user_prompt: str):
        """Append a new user message, and set the corresponding answer of the model to `None`.

        Parameters
        ----------
        user_prompt : str
            The user message.
        """

        if None in self.model_history_text:
            raise ValueError('Cannot append a new user message before the model answered to the previous messages.')

        self.user_history_text.append(user_prompt)
        self.model_history_text.append(None)


    def append_model_message(self, model_output: str):
        """Append a new model message, by modifying the last `None` value in-place. Should always be called after
        `append_user_message`, before a new call to `append_user_message`.

        Parameters
        ----------
        model_output : str
            The model message.
        """

        if self.model_history_text[-1] is None:
            self.model_history_text[-1] = model_output
        else:
            raise ValueError('It looks like the last user message was already answered by the model.')
        

    def append_to_last_model_message(self, model_output: str):
        """Append text to the last model message, in case the `max_new_tokens` was set to a value too low
        to finish the model answer.

        Parameters
        ----------
        model_output : str
            The model message.
        """

        if self.model_history_text[-1] is None:
            raise ValueError('The last user message was never answered. You should use `append_model_message`.')
        else:
            self.model_history_text[-1] += model_output
        

    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # This seems to be the accepted way to treat inputs for conversation with a model that was not specifically
        # fine-tuned for conversation. This is the DialoGPT way of handling conversation, but is in fact reused by
        # all other tokenizers that we use.

        prompt = ''

        for user_message, model_response in self:

            prompt += user_message + self.eos_token
            if model_response is not None:
                prompt += model_response + self.eos_token

        return prompt
    

    def get_last_turn_continuation_prompt(self) -> str:
        """Format the prompt to feed to the model in order to continue the last turn of the model output, in case
        `max_new_tokens` was set to a low value and the model did not finish its output.
        """

        if len(self) == 0:
            raise RuntimeError('Cannot continue the last turn on an empty conversation.')
    
        if self.model_history_text[-1] is None:
            raise RuntimeError('Cannot continue an empty last turn.')
        
        # Use a copy since we will modify the last model turn
        conv_copy = copy.deepcopy(self)
        last_model_output = conv_copy.model_history_text[-1]
        # Set it to None to mimic the behavior of an unanswered turn
        conv_copy.model_history_text[-1] = None

        # Get prompt of conversation without the last model turn
        prompt = conv_copy.get_prompt()
        # Reattach last turn, with or without additional space
        if self.add_space_to_continuation_prompt:
            prompt += ' ' + last_model_output
        else:
            prompt += last_model_output

        return prompt
    

    def get_extra_eos(self) -> list[str]:
        return self.extra_eos_tokens
    

    def erase_conversation(self):
        """Reinitialize the conversation.
        """

        self.user_history_text = []
        self.model_history_text = []

    
    def set_conversation(self, past_user_inputs: list[str], past_model_outputs: list[str]):
        """Set the conversation.
        """

        self.user_history_text = past_user_inputs
        self.model_history_text = past_model_outputs


    def set_few_shot_examples(self, user_few_shot: list[str], model_few_shot: list[str]):
        """Set the few shot examples.
        """

        self.user_few_shot = user_few_shot
        self.model_few_shot = model_few_shot


    def to_gradio_format(self) -> list[list[str, str]]:
        """Convert the current conversation to gradio chatbot format. This does NOT display the few-shot turns.
        """

        if len(self) == 0:
            return [[None, None]]

        return [list(conv_turn) for conv_turn in self.iter_without_few_shot()]
    


# reference: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
class StarChatConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Special tokens
        self.system_token = '<|system|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.sep_token = '<|end|>'

        # extra eos
        self.extra_eos_tokens = [self.sep_token]


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """
        
        prompt = self.system_token + '\n' + self.system_prompt + self.sep_token + '\n' if self.system_prompt != '' else ''

        for user_message, model_response in self:

            prompt += self.user_token + '\n' + user_message + self.sep_token + '\n'
            if model_response is not None:
                prompt += self.assistant_token + '\n' + model_response + self.sep_token + '\n'
            else:
                prompt += self.assistant_token + '\n'

        return prompt
    

# reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L334
class VicunaConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Override value
        self.add_space_to_continuation_prompt = True

        self.system_prompt = ("A chat between a curious user and an artificial intelligence assistant. "
                              "The assistant gives helpful, detailed, and polite answers to the user's questions.")

        self.user_token = 'USER'
        self.assistant_token = 'ASSISTANT'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        prompt = self.system_prompt + ' ' if self.system_prompt != '' else ''

        for user_message, model_response in self:

            prompt += self.user_token + ': ' + user_message + ' '
            if model_response is not None:
                prompt += self.assistant_token + ': ' + model_response + self.eos_token
            else:
                prompt += self.assistant_token + ':'

        return prompt
    
    

# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
class Llama2Conversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Override value
        self.add_space_to_continuation_prompt = True

        self.bos_token = '<s>'

        self.system_prompt = ("You are a helpful, respectful and honest assistant. Always answer as helpfully "
                              "as possible, while being safe. Your answers should not include any harmful, "
                              "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that "
                              "your responses are socially unbiased and positive in nature.\n\n"
                              "If a question does not make any sense, or is not factually coherent, explain why "
                              "instead of answering something not correct. If you don't know the answer to a "
                              "question, please don't share false information.")
        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'

        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # If we are not using system prompt, do not add the template formatting with empty prompt
        if self.system_prompt.strip() != '':
            system_prompt = self.system_template.format(system_prompt=self.system_prompt.strip())
        else:
            system_prompt = ''
        prompt = ''

        for i, (user_message, model_response) in enumerate(self):

            if i == 0:
                # Do not add bos_token here as it will be added automatically at the start of the prompt by 
                # the tokenizer 
                prompt += self.user_token + ' ' + system_prompt + user_message.strip() + ' '
            else:
                prompt += self.bos_token + self.user_token + ' ' + user_message.strip() + ' '
            if model_response is not None:
                prompt += self.assistant_token + ' ' + model_response.strip() + ' ' + self.eos_token
            else:
                prompt += self.assistant_token

        return prompt
    

# references: https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf#chat_prompt
# and https://github.com/facebookresearch/codellama/blob/main/llama/generation.py#L506-L548
class CodeLlama70BConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Override value
        self.add_space_to_continuation_prompt = False

        self.bos_token = '<s>'

        self.system_prompt = ("You are a helpful, respectful and honest assistant. Always answer as helpfully "
                              "as possible, while being safe. Your answers should not include any harmful, "
                              "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that "
                              "your responses are socially unbiased and positive in nature.\n\n"
                              "If a question does not make any sense, or is not factually coherent, explain why "
                              "instead of answering something not correct. If you don't know the answer to a "
                              "question, please don't share false information.")

        self.system_token = 'Source: system'
        self.user_token = 'Source: user'
        self.assistant_token = 'Source: assistant'
        self.separator = '<step>'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # If we are not using system prompt, the original code still add an empty string
        prompt = self.system_token + '\n\n ' + self.system_prompt.strip() + ' ' + self.separator + ' '

        for user_message, model_response in self:

            prompt += self.user_token + '\n\n ' + user_message.strip() + ' ' + self.separator + ' '

            if model_response is not None:
                prompt += self.assistant_token + '\n\n ' + model_response.strip() + ' ' + self.separator + ' '
            else:
                prompt += self.assistant_token + '\nDestination: user\n\n ' 

        return prompt
    

# reference: https://docs.mistral.ai/usage/guardrailing/
class MistralConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Override value
        self.add_space_to_continuation_prompt = True

        self.system_prompt = (
            "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, "
            "unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
        )

        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        system_prompt = self.system_prompt.strip()

        prompt = ''
        for i, (user_message, model_response) in enumerate(self):

            if i == 0:
                prompt += self.user_token + ' ' + system_prompt + ' ' + user_message.strip() + ' '
            else:
                prompt += self.user_token + ' ' + user_message.strip() + ' '
            if model_response is not None:
                prompt += self.assistant_token + ' ' + model_response.strip() + self.eos_token
            else:
                prompt += self.assistant_token

        return prompt
    

# reference: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
class ZephyrConversation(GenericConversation):

    def __init__(self, eos_token: str):

        super().__init__(eos_token)

        # Override value
        self.add_space_to_continuation_prompt = False

        self.system_prompt = (
            "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, "
            "unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
        )

        self.system_token = '<|system|>'
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # If we are not using system prompt, do not add the template formatting with empty prompt

        if self.system_prompt.strip() != '':
            prompt = self.system_token + '\n' + self.system_prompt.strip() + self.eos_token + '\n'
        else:
            prompt = ''

        for user_message, model_response in self:

            prompt += self.user_token + '\n' + user_message.strip() + self.eos_token + '\n'

            if model_response is not None:
                prompt += self.assistant_token + '\n' + model_response.strip() + self.eos_token + '\n'
            else:
                prompt += self.assistant_token + '\n'

        return prompt
    


# Mapping from model name to conversation class name
CONVERSATION_MAPPING = {
    # StarChat
    'star-chat-alpha': StarChatConversation,
    'star-chat-beta': StarChatConversation,

    # Vicuna (1.3)
    'vicuna-7B': VicunaConversation,
    'vicuna-13B': VicunaConversation,

    # Llama2-chat
    'llama2-7B-chat': Llama2Conversation,
    'llama2-13B-chat': Llama2Conversation,
    'llama2-70B-chat': Llama2Conversation,

    # Code-llama-instruct
    'code-llama-7B-instruct': Llama2Conversation,
    'code-llama-13B-instruct': Llama2Conversation,
    'code-llama-34B-instruct': Llama2Conversation,
    # Special syntax for the 70B version
    'code-llama-70B-instruct': CodeLlama70BConversation,

    # Mistral
    'mistral-7B-instruct': MistralConversation,

    # Zephyr
    'zephyr-7B-alpha': ZephyrConversation,
    'zephyr-7B-beta': ZephyrConversation,
}



def get_empty_conversation_template(model_name: str) -> GenericConversation:
    """Return the conversation object corresponding to `model_name`.

    Parameters
    ----------
    model_name : str
        Name of the current model.

    Returns
    -------
    GenericConversation
        A conversation object template class corresponding to `model_name`.

    """

    if model_name not in loader.ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*loader.ALLOWED_MODELS,}.')
    
    # TODO: maybe change this way of obtaining the eos token for a given model as it forces to load the
    # tokenizer for nothing (maybe create a mapping from name to eos?). For now it is sufficient as 
    # loading a tokenizer is sufficiently fast
    tokenizer = loader.load_tokenizer(model_name)
    eos_token = tokenizer.eos_token

    if model_name in CONVERSATION_MAPPING.keys():
        conversation = CONVERSATION_MAPPING[model_name](eos_token=eos_token)
    else:
        conversation = GenericConversation(eos_token=eos_token)

    return conversation


def get_conversation_from_yaml_template(model_name: str, path: str) -> GenericConversation:
    """Return the conversation object corresponding to `model_name`, with attribute set from the yaml
    file at `path`.

    Parameters
    ----------
    model_name : str
        Name of the current model.
    path : str
        Path to the template file.

    Returns
    -------
    GenericConversation
        A conversation object template class corresponding to `model_name`.

    """

    if model_name not in loader.ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*loader.ALLOWED_MODELS,}.')
    
    # TODO: maybe change this way of obtaining the eos token for a given model as it forces to load the
    # tokenizer for nothing (maybe create a mapping from name to eos?). For now it is sufficient as 
    # loading a tokenizer is sufficiently fast
    tokenizer = loader.load_tokenizer(model_name)
    eos_token = tokenizer.eos_token

    if model_name in CONVERSATION_MAPPING.keys():
        conversation = CONVERSATION_MAPPING[model_name].from_yaml_template(path, eos_token=eos_token)
    else:
        conversation = GenericConversation.from_yaml_template(path, eos_token=eos_token)

    return conversation




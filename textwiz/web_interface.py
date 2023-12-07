"""This module provides convenience functions to use when creating a Gradio web app."""
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import gradio as gr

from .generation import HFModel
from .streamer import TextContinuationStreamer
from .conversation_template import GenericConversation


TIMEOUT = 20


def text_generation(model: HFModel, prompt: str, max_new_tokens: int, do_sample: bool, top_k: int, top_p: float,
                    temperature: float, use_seed: bool, seed: int, **kwargs) -> str:
    """Text generation with `model`. This is a generator yielding tokens as they are generated.

    Parameters
    ----------
    model : HFModel
        The model to use for generation.
    prompt : str
        The prompt to the model.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int,
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Yields
    ------
    Iterator[str]
        String containing the sequence generated.
    """
    
    if not use_seed:
        seed = None

    # To show text as it is being generated
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, timeout=TIMEOUT, skip_special_tokens=True)

    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.generate_text, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 top_k=top_k, top_p=top_p, temperature=temperature, seed=seed,
                                 truncate_prompt_from_output=True, streamer=streamer, **kwargs)
    
        # Get results from the streamer and yield it
        try:
            # Ask the streamer to skip prompt and reattach it here to avoid showing special prompt formatting
            generated_text = prompt
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {TIMEOUT} s)')
    
        # Get actual result and yield it (which may be slightly different due to postprocessing)
        generated_text = future.result()
        yield prompt + generated_text



def chat_generation(model: HFModel, conversation: GenericConversation, prompt: str, max_new_tokens: int,
                    do_sample: bool,top_k: int, top_p: float, temperature: float, use_seed: bool, seed: int,
                    **kwargs) -> tuple[str, GenericConversation, list[list]]:
    """Chat generation with `model`. This is a generator yielding tokens as they are generated.

    Parameters
    ----------
    model : HFModel
        The model to use for generation.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    prompt : str
        The prompt to the model.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool, optional
        Whether to use a fixed seed for reproducibility., by default False.
    seed : int
        An optional seed to force the generation to be reproducible.

    Yields
    ------
    Iterator[tuple[str, GenericConversation, list[list]]]
        Correspond to gradio components (prompt, conversation, chatbot).
    """
    
    if not use_seed:
        seed = None

    # To show text as it is being generated
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, timeout=TIMEOUT, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(model.generate_conversation, prompt, conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True,
                                 streamer=streamer, **kwargs)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield '', conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {TIMEOUT} s)')
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield '', conversation, conversation.to_gradio_format()



def continue_generation(model: HFModel, conversation: GenericConversation, additional_max_new_tokens: int,
                        do_sample: bool, top_k: int, top_p: float, temperature: float, use_seed: bool,
                        seed: int, **kwargs) -> tuple[GenericConversation, list[list]]:
    """Continue the last turn of the `model` output. This is a generator yielding tokens as they are generated.

    Parameters
    ----------
    model : HFModel
        The model to use for generation.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    additional_max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Yields
    ------
    Iterator[tuple[GenericConversation, list[list]]]
        Correspond to gradio components (conversation, chatbot).
    """

    if len(conversation) == 0:
        gr.Warning(f'You cannot continue an empty conversation.')
        yield conversation, conversation.to_gradio_format()
        return
    if conversation.model_history_text[-1] is None:
        gr.Warning('You cannot continue an empty last turn.')
        yield conversation, conversation.to_gradio_format()
        return

    if not use_seed:
        seed = None

    # To show text as it is being generated
    streamer = TextContinuationStreamer(model.tokenizer, skip_prompt=True, timeout=TIMEOUT, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(model.continue_last_conversation_turn, conv_history=conversation,
                                 max_new_tokens=additional_max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True, streamer=streamer,
                                 **kwargs)
        
        # Get results from the streamer and yield it
        try:
            generated_text = conv_copy.model_history_text[-1]
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {TIMEOUT} s)')

    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, conversation.to_gradio_format()



def retry_chat_generation(model: HFModel, conversation: GenericConversation, max_new_tokens: int, do_sample: bool,
                          top_k: int, top_p: float, temperature: float, use_seed: bool,
                          seed: int, **kwargs) -> tuple[GenericConversation, list[list]]:
    """Regenerate the last turn of the conversation. This is a generator yielding tokens as they are generated.

    Parameters
    ----------
    model : HFModel
        The model to use for generation.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int
        How many new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    use_seed : bool
        Whether to use a fixed seed for reproducibility.
    seed : int
        An optional seed to force the generation to be reproducible.

    Yields
    ------
    Iterator[tuple[GenericConversation, list[list]]]
        Correspond to gradio components (conversation, chatbot).
    """

    if len(conversation) == 0:
        gr.Warning(f'You cannot retry generation on an empty conversation.')
        yield conversation, conversation.to_gradio_format()
        return
    if conversation.model_history_text[-1] is None:
        gr.Warning('You cannot retry generation on an empty last turn')
        yield conversation, conversation.to_gradio_format()
        return
    
    if not use_seed:
        seed = None

    # Remove last turn
    prompt = conversation.user_history_text[-1]
    _ = conversation.user_history_text.pop(-1)
    _ = conversation.model_history_text.pop(-1)

    # To show text as it is being generated
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, timeout=TIMEOUT, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(model.generate_conversation, prompt, system_prompt=None, conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=seed, truncate_if_conv_too_long=True, streamer=streamer,
                                 **kwargs)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conv_copy, conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {TIMEOUT} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, conversation.to_gradio_format()



def simple_authentication(credentials_file: str, username: str, password: str) -> bool:
    """Simple authentication method.

    Parameters
    ----------
    credentials_file : str
        Path to the credentials.
    username : str
        The username provided.
    password : str
        The password provided.

    Returns
    -------
    bool
        Return True if both the username and password match some credentials stored in `credentials_file`. 
        False otherwise.
    """

    with open(credentials_file, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines() if line.strip() != '']

    valid_usernames = lines[0::2]
    valid_passwords = lines[1::2]

    if username in valid_usernames:
        index = valid_usernames.index(username)
        # Check that the password also matches at the corresponding index
        if password == valid_passwords[index]:
            return True
    
    return False
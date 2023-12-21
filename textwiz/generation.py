import os
import warnings
import gc
import psutil
import math
import copy
import re

import torch
import scipy
import numpy as np
from transformers import StoppingCriteriaList, GenerationConfig

from . import loader
from . import stopping
from . import utils
from .prompt_template import GenericPromptTemplate, get_prompt_template
from .conversation_template import GenericConversation, get_empty_conversation_template, get_conversation_from_yaml_template
from .code_parser import CodeParser
from .constants import SENTENCEPIECE_CHARACTER


class HFModel(object):
    """Class encapsulating a HuggingFace model and its tokenizer to generate text. 
    """

    def __init__(self, model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | str | None = None, gpu_rank: int = 0):
        
        # Save the current allocated memory on each gpu to estimate model size after loading
        if torch.cuda.is_available():
            reference_memory = {}
            for i in range(torch.cuda.device_count()):
                reference_memory[i] = torch.cuda.memory_allocated(i)

        # Actually load the model and tokenizer
        self.model, self.tokenizer = loader.load_model_and_tokenizer(model_name, quantization_8bits=quantization_8bits,
                                                                     quantization_4bits=quantization_4bits, dtype=dtype,
                                                                     max_fraction_gpu_0=max_fraction_gpu_0,
                                                                     max_fraction_gpus=max_fraction_gpus,
                                                                     device_map=device_map, gpu_rank=gpu_rank)
        
        # Compute the memory footprint of the model on each gpu
        self.gpu_memory_map = {}

        # In this case, the model is on multiple devices
        if hasattr(self.model, 'hf_device_map'):
            self.device_map = self.model.hf_device_map

            gpu_devices = set(self.model.hf_device_map.values())
            gpu_devices.discard('cpu')
            gpu_devices.discard('disk')

            # Compute the gpus memory footprint
            self.input_device = min(gpu_devices) if len(gpu_devices) > 0 else 'cpu'
            for device in gpu_devices:
                self.gpu_memory_map[device] = (torch.cuda.memory_allocated(device) - reference_memory[device]) / 1024**3
        
        # In this case, the model is on a single device
        else:
            device = next(self.model.parameters()).get_device()
            self.device_map = 'cpu' if device == -1 else f'cuda:{device}'
            self.input_device = 'cpu' if device == -1 else device

            # Compute the gpu memory if the device is a gpu
            if device != -1:
                self.gpu_memory_map[device] = (torch.cuda.memory_allocated(device) - reference_memory[device]) / 1024**3

        # Maximum memory taken by the model on gpus, or on the cpu
        if len(self.gpu_memory_map) > 0:
            self.max_memory_footprint = max(self.gpu_memory_map.values())
        else:
            # Estimate the footprint via the number of parameters in this case
            self.max_memory_footprint = self.model.get_memory_footprint() / 1024**3

        self.model_name = model_name
        self.quantization_8bits = quantization_8bits
        self.quantization_4bits = quantization_4bits
        # May be different from the dtype given in the arguments so use the model attribute
        self.dtype = self.model.dtype


        # Initialize the prompt template to use 
        self.prompt_template = get_prompt_template(self.model_name)

        # Extra eos tokens
        self.extra_eos_tokens = self.prompt_template.get_extra_eos()

        # Flag to check if the model is a chat model by default
        self._is_chat_model = self.prompt_template.default_mode == 'chat'

    
    def dtype_category(self) -> str:
        """Return a string representation of the model dtype."""
        if self.quantization_4bits:
            return 'int4'
        elif self.quantization_8bits:
            return 'int8'
        else:
            return str(self.dtype).split('.', 1)[1]

    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.model_name}, quantization_8bits={self.quantization_8bits}, '
                f'quantization_4bits={self.quantization_4bits}, dtype={self.dtype})')
    
    
    def __str__(self) -> str:
        return f'{self.model_name}, with dtype {self.dtype_category()}'
        
        
    def is_chat_model(self) -> bool:
        """Check if the model was originally optimized as a chat agent."""
        return self._is_chat_model
    

    def get_gpu_memory_footprint(self) -> dict:
        """Return the memory footprint of the model on each GPU device it uses, in GiB."""
        return copy.deepcopy(self.gpu_memory_map)
    

    def get_memory_footprint(self) -> dict:
        """Return the memory footprint of the model on each device it uses, in GiB. In case of a custom `device_map`
        where both gpu devices AND cpu and/or disk were specified, this function is not accurate.
        """

        gpu_footprint = self.get_gpu_memory_footprint()
        if len(gpu_footprint) == 0:
            return {'cpu': self.max_memory_footprint}
        else:
            # If the custom device map contains both gpu device and cpu (and/or disk), this is not accurate as
            # we only return the footprint of the gpus (computing the footprint of the cpu is hard and not
            # precise, and this case should never appear in practice)
            return gpu_footprint
        

    def get_max_device_memory_footprint(self) -> float:
        """Return the maximum (accross devices) memory used by the model."""
        return self.max_memory_footprint
    

    def get_gpu_devices(self) -> tuple[int]:
        """Return the gpu devices used by the model."""
        return tuple(sorted(self.gpu_memory_map.keys()))


    def format_prompt(self, prompt: str, model_context: str = '', infill_suffix: str = '', system_prompt: str = '',
                      prompt_template_mode: str = 'default') -> str:
        """Format the prompt according to the prompt template.

        Parameters
        ----------
        prompt : str
            The prompt to the model.
        model_context : str, optional
            An optional context forming the start of the model answer. For `generation` mode, this is simply
            appended to `prompt`. By default ''.
        infill_suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all `prompt_template_mode` except
            `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all 
            `prompt_template_mode` except `chat`, by default ''.
        prompt_template_mode: str
            The template mode for formatting the prompt. One of `('default', 'generation', 'infill', 'chat')`.
            Note that changing this value may result in errors or inconsistent results as usually a model is
            optimized for only one given prompt format. By default 'default', which chooses the best mode for
            the current model.

        Returns
        -------
        str
            The formatted prompt to use in the model forward.
        """
        
        # Set the template mode
        old_mode = self.prompt_template.mode
        self.prompt_template.set_mode(prompt_template_mode)

        formatted_prompt = self.prompt_template.get_prompt(prompt, model_context=model_context, suffix=infill_suffix,
                                                           system_prompt=system_prompt)
        
        # Reset template mode to previous mode
        self.prompt_template.set_mode(old_mode)

        return formatted_prompt
    

    def create_generation_config(self, max_new_tokens: int, min_new_tokens: int, do_sample: bool,
                                 top_k: int | None, top_p: float | None, temperature: float) -> GenerationConfig:
        """Create a new `GenerationConfig` object to pass to `model.generate()` to control the generation strategy.
        This is needed because by default `generate()` uses `self.model.generation_config` if the `generation_config`
        parameter is not provided, which may conflict with some of our parameters and thus provide incorrect
        or suprising results.
        
        Parameters
        ----------
        max_new_tokens : int
            How many new tokens to generate.
        min_new_tokens : int
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS in some cases.
        do_sample : bool
            Whether to introduce randomness in the generation.
        top_k : int | None
            How many tokens with max probability to consider for random sampling. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None
            The probability density covering the new tokens to consider for random sampling. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness). Passing 0 is equivalent to setting `do_sample=False`.

        Returns
        -------
        GenerationConfig
            Config which controls the text generation.
        """

        # Setting the temperature to 0 is equivalent to greedy search thus we explicitly set do_sample=False
        if temperature == 0:
            do_sample = False

        # Retrieve eos_token_id (note that the attribute exists in all cases)
        if self.model.generation_config.eos_token_id is not None:
            eos_token_id = self.model.generation_config.eos_token_id
        elif self.model.config.eos_token_id is not None:
            eos_token_id = self.model.config.eos_token_id
        elif self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id
        else:
            raise RuntimeError('Impossible to find the `eos_token_id`.')

        # Retrieve bos_token_id (note that the attribute exists in all cases)
        if self.model.generation_config.bos_token_id is not None:
            bos_token_id = self.model.generation_config.bos_token_id
        elif self.model.config.bos_token_id is not None:
            bos_token_id = self.model.config.bos_token_id
        elif self.tokenizer.bos_token_id is not None:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            raise RuntimeError('Impossible to find the `bos_token_id`.')
        
        # Retrieve pad_token_id and set it to eos_token_id if it does not exist (note that the attribute
        # exists in all cases)
        if self.model.generation_config.pad_token_id is not None:
            pad_token_id = self.model.generation_config.pad_token_id
        elif self.model.config.pad_token_id is not None:
            pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            # We don't really need a padding token in our case as we never need to pad, we just make sure
            # it is a known special token (eos token) that will be generated when the sequence is finished.
            # This way it is automatically removed from the sequence when using `decode(..., skip_special_tokens=True)`
            pad_token_id = eos_token_id

        # create the config
        generation_config = GenerationConfig(eos_token_id=eos_token_id, bos_token_id=bos_token_id,
                                             pad_token_id=pad_token_id, max_new_tokens=max_new_tokens,
                                             min_new_tokens=min_new_tokens, do_sample=do_sample)
        
        # Add parameters to the config
        if do_sample:
            unused = generation_config.update(top_k=top_k, top_p=top_p, temperature=temperature)
            assert len(unused) == 0, 'There is a typo in some generation config parameters.'

        return generation_config


    def generate_text(
            self,
            prompt: str,
            model_context: str = '',
            infill_suffix: str = '',
            system_prompt: str = '',
            prompt_template_mode: str = 'default',
            max_new_tokens: int = 256,
            min_new_tokens: int = 0,
            do_sample: bool = True,
            top_k: int | None = 50,
            top_p: float | None = 0.90,
            temperature: float = 0.8,
            num_return_sequences: int = 1,
            batch_size: int | None = None,
            seed: int | None = None,
            stopping_patterns: stopping.StoppingType | list[str] | tuple[str] | re.Pattern | str | None = None,
            parser: CodeParser | None = None,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str | list[str]:
        """Generate text according to `prompt` using the parameters specified.

        Prompt formatting parameters
        ----------------------------

        prompt : str
            The prompt to the model.
        model_context : str, optional
            An optional context forming the start of the model answer. For `generation` mode, this is simply
            appended to `prompt`. By default ''.
        infill_suffix : str, optional
            An optional suffix to form the prompt. This is ignored for all `prompt_template_mode` except
            `infill`, by default ''.
        system_prompt : str, optional
            An optional system prompt to append at the beginning for chat mode. This is ignored for all 
            `prompt_template_mode` except `chat`, by default ''.
        prompt_template_mode: str
            The template mode for formatting the prompt. One of `('default', 'generation', 'infill', 'chat')`.
            Note that changing this value may result in errors or inconsistent results as usually a model is
            optimized for only one given prompt format. By default 'default', which chooses the best mode for
            the current model.

        Generation parameters
        ---------------------
        
        max_new_tokens : int, optional
            How many new tokens to generate, by default 256.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 0.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.8. Passing 0 is equivalent to setting `do_sample=False`.
        num_return_sequences : int, optional
            How many sequences to generate according to the `prompt`, by default 1.
        batch_size : int | None, optional
            Max batch size for the model forward pass, in case `num_return_sequences` is large. If `None`, will
            try to determine the largest possible batch size that does not result in memory error. By default None.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        stopping_patterns : StoppingType | list[str] | tuple[str] | re.Pattern | str | None, optional
            The type of early stopping to use. This should be an instance of the `StoppingType` enum, or eventually
            a list or tuple of str, in which case the iterable will be passed to a `TextPatternStopping` instance. It can
            also be a re.Pattern or str, which is interpreted as a regex and is passed to a `RegexPatternStopping` instance.
            If `None`, only the `extra_eos_tokens` will be used for early stopping. By default `None`.
        parser: CodeParser | None, optional
            A parser to extract code from generated sequences. The final outputs will only consist of the parsed
            sequences if `post_process_output` is True. Also, the `stopping_patterns` will be applied on the
            parsed sequences. This should be used with caution, as it was designed only for chat models that
            embed code in their output in natural language. The default is None, i.e. no parsing.

        Output formatting parameters
        ----------------------------

        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default True.
        post_process_output : bool, optional
            Whether to post-process the outputs, i.e. truncate according to the `stopping_patterns`. This is
            needed to correctly truncate all sequences if `num_return_sequences > 1`. By default True.

        Returns
        -------
        str | list[str]
            Str containing the generated sequence, or list[str] if `num_return_sequences` > 1.
        """
    
        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = self.create_generation_config(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                          do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                          temperature=temperature)

        if num_return_sequences > 1 and not generation_config.do_sample:
            warnings.warn(('You provided `do_sample=False` or `temperature=0`, i.e. greedy search generation, '
                           'but with `num_return_sequences>1`. All sequences will be identical with greedy search, '
                           'so we explicitly set `num_return_sequences=1`'))
            num_return_sequences = 1

        # Prompt formatting
        formatted_prompt = self.format_prompt(prompt, model_context=model_context, infill_suffix=infill_suffix,
                                              system_prompt=system_prompt, prompt_template_mode=prompt_template_mode)
        
        # Prompt to reattach to output if `truncate_prompt_from_output` is False. This way we reattach the
        # prompt given directly by the user, and not the prompt formatted with potential keywords in all
        # but the most complicated cases
        if infill_suffix == '' and system_prompt == '':
            original_prompt = prompt + model_context
        else:
            original_prompt = formatted_prompt

        # Tokenize the prompt
        input = self.tokenizer.encode(formatted_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria
        stopping_criteria = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns,
                                                              self.extra_eos_tokens, parser)

        # Infer batch size if not given
        if batch_size is None:
            batch_size = self.infer_best_batch_size(input_length, max_new_tokens, num_return_sequences)

        # Anything larger than `num_return_sequences` is useless
        batch_size = min(batch_size, num_return_sequences)

        # This will lower the batch size if needed, in case of possible OOM. This allows to continue without crashing,
        # by reducing the batch size automatically
        first_output, batch_size = self.oom_safe_batch_generation(input, generation_config=generation_config,
                                                                  stopping_criteria=stopping_criteria,
                                                                  batch_size=batch_size,
                                                                  **kwargs)

        # If we require more sequences than the allowed batch size, we need to split the generation into
        # multiple passes
        if num_return_sequences > batch_size:
            batch_sizes = [batch_size]*(num_return_sequences // batch_size)
            remainder = num_return_sequences % batch_size
            if remainder != 0:
                batch_sizes += [remainder]
            assert sum(batch_sizes) == num_return_sequences
        else:
            batch_sizes = [num_return_sequences]

        generated_text = []

        for i, size in enumerate(batch_sizes):

            # Do not recompute the first batch of outputs
            if i == 0:
                outputs = first_output
            else:
                outputs = self.model.generate(input, generation_config=generation_config,
                                              stopping_criteria=stopping_criteria, num_return_sequences=size,
                                              **kwargs)
                
            # Truncate the prompt from the output
            truncated_outputs = outputs[:, input_length:]

            # Post-process the sequences according to stopping patterns and extra eos
            if post_process_output:
                generated_batch = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns,
                                                                  self.extra_eos_tokens, parser)
            else:
                generated_batch = self.tokenizer.batch_decode(truncated_outputs, skip_special_tokens=True)
            
            # reattach the prompt if needed
            if not truncate_prompt_from_output:
                generated_batch = [original_prompt + sequence for sequence in generated_batch]
                # generated_batch = [formatted_prompt + sequence for sequence in generated_batch]
            
            generated_text += generated_batch

        # In this case return a str instead of list[str]
        if num_return_sequences == 1:
            generated_text = generated_text[0]

        return generated_text
    

    @utils.copy_docstring_and_signature(generate_text)
    def __call__(self, *args, **kwargs):
        return self.generate_text(*args, **kwargs)
    

    def infer_best_batch_size(self, input_size: int, max_new_tokens: int, num_return_sequences: int) -> int:
        """Try to infer the best (largest) possible batch size for the model given the current `input_size`,
        and `max_new_tokens`. By default, this function checks if a batch memory footprint estimation exists
        in the folder `memory_estimator`, and falls back to simple heuristics if this is not the case.

        Parameters
        ----------
        input_size : int
            The input length.
        max_new_tokens : int
            The number of tokens to generate.
        num_return_sequences : int
            The number of sequences to generate.
        Returns
        -------
        int
            Estimation of the largest possible batch size.
        """
    
        if not torch.cuda.is_available():
            memory = psutil.virtual_memory().total / 1024**3
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Only take 0.85 of the gpu memory into account in order to not completely clutter the memory
        available_memory = memory*0.85 - self.get_max_device_memory_footprint()

        # Try loading estimator file
        try:
            reference_file = os.path.join(utils.DATA_FOLDER, 'memory_estimator', self.model_name, f'{self.dtype_category()}.json')
            batch_footprint = utils.load_json(reference_file)
            only_scale_with_input_size = batch_footprint.pop('only_scale_with_input_size', False)
            # Convert keys to int
            batch_footprint = {int(k): v for k, v in batch_footprint.items()}
        # If no precise estimate exist, fall back to simple heuristics
        except FileNotFoundError:
            return self.infer_best_batch_size_by_heuristics(available_memory)


        x = list(batch_footprint.keys())
        y = list(batch_footprint.values())
        # sort according to increasing sequence
        sorting = np.argsort(x)
        x = np.array(x)[sorting]
        y = np.array(y)[sorting]
        
        # Memory usage is linear wrt to sequence length when using K-V cache
        fit = scipy.stats.linregress(x, y)
        intercept = fit.intercept
        slope = fit.slope
        r2 = fit.rvalue**2

        # If the flag `only_scale_with_input_size` is active, the memory needed for subsequent forward passes
        # is negligible compared to the memory needed to compute the K-V cache the first time
        sequence_length = input_size if only_scale_with_input_size else input_size + max_new_tokens

        # This should always be the case, but check it if for some reason the behavior is not sufficiently linear
        if r2 >= 0.95:
            memory_needed = intercept + slope * sequence_length
        # In this case, fall back to simple heuristics
        else:
            return self.infer_best_batch_size_by_heuristics(available_memory)

        return int(available_memory // memory_needed)
    

    def infer_best_batch_size_by_heuristics(self, available_memory: float) -> int:
        """Infer the largest possible batch size using very simple and raw heuristics. It only uses the number
        of parameters of the model, which is not a very good indicator. 

        Parameters
        ----------
        available_memory : float
            The memory available for the forward pass.

        Returns
        -------
        int
            A very raw estimate of the best batch size.
        """

        parameters = self.parameters_count()
        if parameters < 5:
            batch = int(available_memory // 0.5)
        elif parameters < 10:
            batch = int(available_memory // 1)
        elif parameters < 20:
            batch = int(available_memory // 2)
        else:
            batch = int(available_memory // 3)
        
        return max(batch, 1)


    def oom_safe_batch_generation(self, input: torch.Tensor, generation_config: GenerationConfig,
                                  stopping_criteria: StoppingCriteriaList | None, batch_size: int,
                                  **kwargs) -> tuple[torch.Tensor, int]:
        """Generate text by recursively recovering from possible memory errors (OOMs) by lowering the batch size.
        Note that it is not possible to retry immediately in the except block because the exception retains the
        tensors already allocated in the try block which causes an immediate new OOM
        (see https://github.com/pytorch/pytorch/issues/18853)
        """
        retry = False

        # Try generating result
        try:
            out = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      num_return_sequences=batch_size, **kwargs)
        
        except RuntimeError as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                retry = True
            else:
                raise e

        if retry:
            if batch_size == 1:
                raise RuntimeError('Even a batch size of 1 causes an OOM. Cannot generate with current config.')
            new_batch_size = max(1, math.floor(batch_size*0.8))
            warnings.warn(f'Reducing batch size from {batch_size} to {new_batch_size} due to memory overflow (OOM).', RuntimeWarning)
            gc.collect()
            torch.cuda.empty_cache()
            return self.oom_safe_batch_generation(input, generation_config=generation_config,
                                                  stopping_criteria=stopping_criteria, batch_size=new_batch_size,
                                                  **kwargs)
        else:
            return out, batch_size
        

    def parameters_count(self) -> float:
        """Return the (approximate) number of parameters of the current model, in billions.
        Note that shared parameters will be counted twice by this current function, thus it is only approximate.

        Returns
        -------
        float
            The number of parameters, in billions.
        """

        return sum(map(torch.numel, self.model.parameters())) / 1e9
    

    def set_prompt_template(self, template: GenericPromptTemplate):
        """Set the prompt template."""
        self.prompt_template = template


    def generate_conversation(
            self,
            prompt: str,
            system_prompt: str | None = None,
            conv_history: GenericConversation | None = None,
            max_new_tokens: int = 512,
            min_new_tokens: int = 0,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.90,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: stopping.StoppingType | list[str] | tuple[str] | re.Pattern | str | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> GenericConversation:
        """Generate a conversation turn between a user and the model, according to new user input `prompt`.

        Input parameters
        ----------------
        prompt : str
            The new prompt of the user to the model.
        system_prompt : str | None
            An optional system prompt to guide the style of the model answers. The default is `None` which uses
            the system prompt of the current `conv_history`, or the default one if `conv_history` is not provided.
        conv_history : GenericConversation | None
            An optional existing conversation object, representing the current dialogue between the user and
            the model.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            How many new tokens to generate, by default 512.
        min_new_tokens : int, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
            force the model to generate an output, instead of immediately generating EOS, by default 0.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.8. Passing 0 is equivalent to setting `do_sample=False`.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        stopping_patterns : StoppingType | list[str] | tuple[str] | re.Pattern | str | None, optional
            The type of early stopping to use. This should be an instance of the `StoppingType` enum, or eventually
            a list or tuple of str, in which case the iterable will be passed to a `TextPatternStopping` instance. It can
            also be a re.Pattern or str, which is interpreted as a regex and is passed to a `RegexPatternStopping` instance.
            If `None`, only the `extra_eos_tokens` will be used for early stopping. By default `None`.
        truncate_if_conv_too_long : bool, optional
            Whether to truncate the conversation history if it becomes larger than the model maximum capacity,
            by default True.

        Returns
        -------
        GenericConversation
            A conversation object, with the dialogue history updated with the current turn.
        """

        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = self.create_generation_config(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                                          do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                          temperature=temperature)

        # Check that the history is not empty
        if conv_history is None:
            conv_history = self.get_empty_conversation()

        # Set system prompt
        if system_prompt is not None:
            conv_history.set_system_prompt(system_prompt)

        # Add the prompt to the current conversation
        conv_history.append_user_message(prompt)

        # Generate and tokenize the full prompt
        if truncate_if_conv_too_long:
            truncated_conv = self.truncate_conversation(conv_history, max_new_tokens, continuation=False)
            full_prompt = truncated_conv.get_prompt()
        else:
            full_prompt = conv_history.get_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria in case the model has some extra eos tokens to process
        stopping_criteria  = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns,
                                                               self.extra_eos_tokens, None)

        outputs = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      num_return_sequences=1, **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # Post-process the sequences according to potential extra eos tokens
        response = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns=stopping_patterns,
                                                   extra_eos_tokens=self.extra_eos_tokens)
        
        # Append output to the conv
        conv_history.append_model_message(response[0])

        return conv_history
    


    def continue_last_conversation_turn(
            self,
            conv_history: GenericConversation,
            max_new_tokens: int = 128,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.90,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: stopping.StoppingType | list[str] | tuple[str] | re.Pattern | str | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> GenericConversation:
        """Continue the last conversation turn if the model stopped too early due to `max_new_tokens` being too
        low.

        Input parameters
        ----------------
        conv_history : GenericConversation
            An existing conversation object, representing the current dialogue between the user and
            the model.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            How many new tokens to generate, by default 128.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.8. Passing 0 is equivalent to setting `do_sample=False`.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible.
        stopping_patterns : StoppingType | list[str] | tuple[str] | re.Pattern | str | None, optional
            The type of early stopping to use. This should be an instance of the `StoppingType` enum, or eventually
            a list or tuple of str, in which case the iterable will be passed to a `TextPatternStopping` instance. It can
            also be a re.Pattern or str, which is interpreted as a regex and is passed to a `RegexPatternStopping` instance.
            If `None`, only the `extra_eos_tokens` will be used for early stopping. By default `None`.
        truncate_if_conv_too_long : bool, optional
            Whether to truncate the conversation history if it becomes larger than the model maximum capacity,
            by default True.

        Returns
        -------
        GenericConversation
            A conversation object, with the dialogue history updated with the current turn.
        """

        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = self.create_generation_config(max_new_tokens=max_new_tokens, min_new_tokens=0,
                                                          do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                          temperature=temperature)

        # Generate and tokenize the full prompt
        if truncate_if_conv_too_long:
            truncated_conv = self.truncate_conversation(conv_history, max_new_tokens, continuation=True)
            full_prompt = truncated_conv.get_last_turn_continuation_prompt()
        else:
            full_prompt = conv_history.get_last_turn_continuation_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria in case the model has some extra eos tokens to process
        stopping_criteria  = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns,
                                                               self.extra_eos_tokens, None)

        outputs = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      num_return_sequences=1, **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # TODO: Maybe find better way to make up for the spaces. Other model should in general never generate
        # this token, so it is still a relatively safe way to do it
        first_token = self.tokenizer.convert_ids_to_tokens(int(truncated_outputs[0, 0]))
        if first_token.startswith(SENTENCEPIECE_CHARACTER):
            add_space = True
        else:
            add_space = False

        # Post-process the sequences according to potential extra eos tokens
        response = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns=stopping_patterns,
                                                   extra_eos_tokens=self.extra_eos_tokens)
        
        # Append output to the conv
        if add_space:
            conv_history.append_to_last_model_message(' ' + response[0])
        else:
            conv_history.append_to_last_model_message(response[0])

        return conv_history


    def get_empty_conversation(self) -> GenericConversation:
        """Return a new empty conversation with the template of the current model."""
        return get_empty_conversation_template(self.model_name)
    

    def get_conversation_from_yaml_template(self, path: str) -> GenericConversation:
        """Return a new conversation from the given yaml attributes (system prompt and few-shot examples)."""
        return get_conversation_from_yaml_template(self.model_name, path)
    

    def get_context_size(self) -> int:
        """Return the maximum context size for the current model."""
        return loader.get_model_context_size(self.model_name)
   

    def truncate_conversation(self, conversation: GenericConversation, max_new_tokens: int,
                              continuation: bool = False) -> GenericConversation:
        """Truncate the current conversation by removing the oldest messages so that the length of the prompt
        + the `max_new_tokens` fit the maximum context length that the model can handle.

        Parameters
        ----------
        conversation : GenericConversation
            The current conversation.
        max_new_tokens : int
            How many new tokens to generate.
        continuation : bool, optional
            Whether we continue the last conversation turn, or create a new one. By default `False`.

        Returns
        -------
        GenericConversation
            The truncated conversation.
        """

        if len(conversation) == 0:
            raise ValueError('Cannot truncate an empty conversation.')
        
        context_size = self.get_context_size()

        new_conv = copy.deepcopy(conversation)
        if continuation:
            full_prompt = new_conv.get_last_turn_continuation_prompt()
        else:
            full_prompt = new_conv.get_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]

        while input_length + max_new_tokens >= context_size:
            # Delete the first actual turn (we keep the optional few-shot turns)
            del new_conv.user_history_text[0]
            del new_conv.model_history_text[0]

            if len(new_conv) == 0:
                raise RuntimeError('The entire conversation got truncated to fit the context size.')

            if continuation:
                full_prompt = new_conv.get_last_turn_continuation_prompt()
            else:
                full_prompt = new_conv.get_prompt()
            input = self.tokenizer.encode(full_prompt, return_tensors='pt')
            input_length = input.shape[-1]

        return new_conv
    

    def perplexity(self, text: str, stride = 512) -> float:
        """Compute the perplexity of given `text`. If the number of tokens is larger than the maximum context size,
        use a sliding window with given `stride`. That is, we will move the input of `stride` tokens at each iteration.
        Thus, the model will always have a context of `max_context_size - stride` in order to compute the negative
        log-likelihood of `stride` new tokens after the first iteration. Small `stride` will give better results but
        will require more forward passes.

        Parameters
        ----------
        text : str
            Text for which to compute perplexity.
        stride : int, optional
            Sliding window parameter, by default 512

        Returns
        -------
        float
            The perplexity of `text` given the current model.
        """

        encoding = self.tokenizer(text, return_tensors='pt')

        max_length = self.get_context_size()
        seq_len = encoding.input_ids.shape[-1]

        if stride >= max_length:
            raise RuntimeError('The stride should be lower than the model maximum context size.')

        # Use reduction='sum' instead of 'mean' to compute correct mean at the end (the mean of the mean is incorrect)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        loss = torch.tensor(0., requires_grad=False, device=self.input_device)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            # Will always be equal to stride except on last iteration
            target_length = end_loc - prev_end_loc

            # Compute inputs and targets
            input_ids = encoding.input_ids[:, begin_loc:end_loc].to(self.input_device)
            target_ids = input_ids.clone()

            # This will mask the (max_length - stride) already processed targets in the loss
            target_ids[:, :-target_length] = -100

            # Remove first target as we cannot compute the probability distribution for the first token of the input.
            # This is not an issue since for first iteration the first token is <BOS>, and it is masked for other iterations
            target_ids = target_ids[:, 1:]
            # Remove batch dimension of size 1 (empty)
            target_ids = target_ids.squeeze(0)

            with torch.no_grad():
                outputs = self.model(input_ids)

                # Extract the logits for all tokens except the last one (we do not care about the probability
                # distribution of what would be the new token if we were performing auto-regresive generation)
                logits = outputs.logits[:, :-1, :]
                # Remove batch dimension of size 1 (empty)
                logits = logits.squeeze(0)

                # Logits now have dimension (len(input_ids)-1, vocab_size). This correspond to the logit distribution
                # for each token given the previous ones. Instead of applying a softmax, taking the probability
                # corresponding to the input token, and summing, we can directly use the CrossEntropyLoss as a trick. 
                # That is, we see it as the loss for a problem with C=vocab_size classes, and an "artificial batch"
                # of size len(input_ids)-1
                loss += criterion(logits, target_ids)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        # Don't forget to apply the exponential after dividing by the total size of the sequence
        perplexity_output = torch.exp(loss / (seq_len-1))

        return perplexity_output.item()
        




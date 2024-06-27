import os
import psutil
import warnings

import torch
import numpy as np

from .base import HFBaseModel
from .. import loader
from ..helpers import utils


class HFEmbeddingModel(HFBaseModel):
    """Class encapsulating a HuggingFace model and its tokenizer to create text embedding. 
    """

    def __init__(self, model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | str | None = None, gpu_rank: int = 0):
        
        # Check name against only embedding models
        loader.check_model_name(model_name, loader.ALLOWED_EMBEDDING_MODELS)
        
        super().__init__(model_name, quantization_8bits, quantization_4bits, dtype, max_fraction_gpu_0,
                         max_fraction_gpus, device_map, gpu_rank)

    
    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract hidden state for the last token, given potential padding.

        Parameters
        ----------
        last_hidden_states : torch.Tensor
            Hidden states of the last layer for all tokens.
        attention_mask : torch.Tensor
            Attention mask of the input.

        Returns
        -------
        torch.Tensor
            Hidden state for the last token.
        """

        # Check if padding is left or right
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

        if left_padding:
            return last_hidden_states[:, -1, :]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            device = last_hidden_states.device
            return last_hidden_states[torch.arange(batch_size, device=device), sequence_lengths, :]

    
    @torch.no_grad()
    def embed(self, inputs: list[str] | str, max_batch_size: int | None = None) -> np.ndarray:
        """Return the embeddings for the given `inputs`.

        Parameters
        ----------
        inputs : list[str] | str
            The input text (or batch of texts)
        max_batch_size : int | None, optional
            If given, will use this as the maximum batch size for a single model pass. By default `None`, i.e
            the maximum will be automatically determined.

        Returns
        -------
        np.ndarray
            The embeddings.
        """

        # Tokenize a first time to check input length
        input_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
        # In this case raise error
        if input_dict['input_ids'].shape[1] > self.get_context_size():
            raise ValueError(('At least one of the inputs is longer than the maximum allowed size. Truncate it in '
                              'smaller chunks, then embed those.'))
        
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        input_length = input_ids.shape[0]

        if torch.cuda.is_available():
            input_ids = input_ids.to(device=self.input_device)
            attention_mask = attention_mask.to(device=self.input_device)

        # Here we assume that chunks were chosen by number of tokens, i.e. all `inputs` have appromately the same length.
        # If this is not the case, then for a large number of ìnputs` it would be better to first divide them by length,
        # and then estimate the batch size BEFORE tokenizing them with padding to maximize efficiency. For example, if only
        # one input is very large compared to the others, then all will be padded to that length so the maximum batch size
        # will be smaller than what would be feasible without that long input.
        if max_batch_size is None:
            max_batch_size = self.infer_best_batch_size(input_ids.shape[1])

        current_index = 0
        final_output = []
        while True:
            
            batch_size = min(max_batch_size, input_length-current_index)
            _slice = slice(current_index, current_index+batch_size)

            # Gradients were already disabled here
            outputs = self.model(input_ids=input_ids[_slice, :], attention_mask=attention_mask[_slice, :])
            embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask[_slice, :])

            # Add current batch to final result
            final_output.append(embeddings.cpu())

            current_index += batch_size
            if current_index >= input_length:
                break

        # Concatenate all result tensors
        final_output = torch.cat(final_output, dim=0).float()

        return final_output.numpy()
    

    @utils.copy_docstring_and_signature(embed)
    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
    

    def infer_best_batch_size(self, input_size: int) -> int:
        """Try to infer the best (largest) possible batch size for the model given the current `input_size`,
        By default, this function checks if a batch memory footprint estimation exists
        in the folder `memory_estimator`, and falls back to simple heuristics if this is not the case.

        Parameters
        ----------
        input_size : int
            The input length.
        Returns
        -------
        int
            Estimation of the largest possible batch size.
        """
    
        # Only take 0.92 of the gpu memory into account in order to not completely clutter the memory
        if not torch.cuda.is_available():
            memory = psutil.virtual_memory().total / 1024**3
            available_memory = memory*0.92 - self.get_max_device_memory_footprint()
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory = memory*0.92 - max(torch.cuda.memory_allocated(device) / 1024**3 for device in self.get_gpu_devices())

        # Try to estimate the memory needed for current inputs
        try:
            reference_file = os.path.join(utils.DATA_FOLDER, 'memory_estimator', 'embedding', self.model_name, f'{self.dtype_category()}.json')
            memory_needed, passes_r2_test = utils.memory_estimation_embedding(reference_file, input_size)
        # If no precise estimate exist, fall back to simple heuristics
        except FileNotFoundError:
            return self.infer_best_batch_size_by_heuristics(available_memory, input_size)

        if not passes_r2_test:
            warnings.warn((f'Memory estimation for {self.model.__class__.__name__} is not sufficiently precise. '
                            'Falling back to heuristics.'))
            return self.infer_best_batch_size_by_heuristics(available_memory, input_size)
        
        return int(available_memory // memory_needed)
    

    def infer_best_batch_size_by_heuristics(self, available_memory: float, input_size: int) -> int:
        """Infer the largest possible batch size using very simple and raw heuristics. It only uses the number
        of parameters of the model, which is not a very good indicator. 

        Parameters
        ----------
        available_memory : float
            The memory available for the forward pass.
        input_size : int
            The input length.

        Returns
        -------
        int
            A very raw estimate of the best batch size.
        """

        parameters = self.parameters_count()
        chunks = (input_size // 4096) + 1 if input_size % 4096 != 0 else input_size // 4096
        if parameters < 5:
            batch = 4 * int(available_memory // chunks)
        elif parameters < 10:
            batch = 2 * int(available_memory // chunks)
        elif parameters < 20:
            batch = int(available_memory // chunks)
        else:
            batch = int(available_memory // (2 * chunks))
        
        return max(batch, 1)
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
        
        if model_name not in loader.ALLOWED_EMBEDDING_MODELS:
            raise ValueError(f'The model name must be one of {*loader.ALLOWED_EMBEDDING_MODELS,}.')
        
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

    
    def embed(self, inputs: list[str] | str, instruction: str | None = None,
              max_batch_size: int | None = None) -> np.ndarray:
        """Return the embeddings for the given `inputs`.

        Parameters
        ----------
        inputs : list[str] | str
            The input text (or batch of texts)
        instruction : str | None, optional
            _description_, by default None
        max_batch_size : int | None, optional
            If given, will use this as the maximum batch size for a single model pass. By default `None`, i.e
            no maximum.

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

        # No maximum -> will pass all inputs as a single batch
        if max_batch_size is None:
            max_batch_size = input_length

        current_index = 0
        final_output = torch.tensor([], dtype=torch.float32, device='cpu', requires_grad=False)
        while True:
            
            batch_size = min(max_batch_size, input_length-current_index)
            _slice = slice(current_index, current_index+batch_size)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids[_slice, :], attention_mask=attention_mask[_slice, :])
                embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask[_slice, :])

            # Concatenate current batch to final vector
            final_output = torch.cat([final_output, embeddings.cpu()], dim=0)

            current_index += batch_size
            if current_index >= input_length:
                break

        return final_output.numpy()
    

    @utils.copy_docstring_and_signature(embed)
    def __call__(self, *args, **kwargs):
        return self.embed(*args, **kwargs)
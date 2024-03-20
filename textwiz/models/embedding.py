import torch
import numpy as np

from .base import HFBaseModel
from .. import loader


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
            return last_hidden_states[:, sequence_lengths, :]

    
    def embed(self, inputs: list[str] | str, instruction: str | None = None) -> np.ndarray:

        # TODO: assert input size is small enough to not truncate
        input_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")

        if torch.cuda.is_available():
            input_dict['input_ids'] = input_dict['input_ids'].cuda()
            input_dict['attention_mask'] = input_dict['attention_mask'].cuda()

        with torch.no_grad():
            outputs = self.model(**input_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, input_dict['attention_mask'])

        return embeddings.cpu().numpy()
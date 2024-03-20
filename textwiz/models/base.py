import copy

import torch

from .. import loader


class HFBaseModel(object):
    """Class encapsulating a HuggingFace model and its tokenizer. Will be used as a super class for all different
    type of models (causal language generation, embedding, etc...)
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


    def parameters_count(self) -> float:
        """Return the (approximate) number of parameters of the current model, in billions.
        Note that shared parameters will be counted twice by this current function, thus it is only approximate.

        Returns
        -------
        float
            The number of parameters, in billions.
        """

        return sum(map(torch.numel, self.model.parameters())) / 1e9
    

    def get_context_size(self) -> int:
        """Return the maximum context size for the current model."""
        return loader.get_model_context_size(self.model_name)
    

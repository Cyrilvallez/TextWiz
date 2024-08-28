import unittest
import os

import torch
import numpy as np

from textwiz import HFCausalModel, PythonParser
from textwiz.helpers import utils

# Cuda is available and we have a big enough GPU
_is_cuda_available = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1024**3) > 15
_default_causal_model_name = 'zephyr-7B-beta'

def require_gpu(test_case):
    """Decorator marking a test that requires CUDA."""
    return unittest.skipUnless(_is_cuda_available, "test requires CUDA")(test_case)

_TEST_PARSER_INPUTS = [
    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)
```""",

    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """    print('test)
    return True""",

    """
    return True
foobar
def test():
    print('test')
should not be code
    """,

    """    print('bar')
foobar
```python
def foo(bar):
    print('bar')
```""",

    # Without backticks
    """Here's a Python implementation of the function:

def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations
    
Here's an example usage:

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",
]


_EXPECTED_PARSER_OUTPUTS = [
    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """    print('test)
    return True""",

    """
    return True

def test():
    print('test')""",

    """    print('bar')

def foo(bar):
    print('bar')""",

    """def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",

]


class UtilitiesTests(unittest.TestCase):

    @require_gpu
    def test_causal_memory_estimation(self):
        model = HFCausalModel(_default_causal_model_name)
        dummy_long_input = torch.randint(0, 5000, (1, 500), device=model.input_device)
        new_tokens = 200

        gpus = model.get_gpu_devices()
        actual_peaks = {}
        for gpu_rank in gpus:
            torch.cuda.reset_peak_memory_stats(gpu_rank)
            actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

        foo = model.model.generate(dummy_long_input, do_sample=False, max_new_tokens=new_tokens, min_new_tokens=new_tokens)

        memory_used = {}
        for gpu_rank in gpus:
            memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
        
        memory_peak = max(memory_used.values())

        version_ = "old" if utils._is_old_version else "new"
        _default_estimation_path = os.path.join(utils.DATA_FOLDER, 'memory_estimator', version_, 'causal', model.model_name, f'{model.dtype_category()}.json')
        memory_estimation, valid_estimation = utils.memory_estimation_causal(_default_estimation_path, dummy_long_input.shape[1], new_tokens)
        self.assertTrue(valid_estimation)

        relative_error = np.abs(memory_peak - memory_estimation) / memory_peak
        print(relative_error)
        # Assert we get an approximation within 2%
        self.assertTrue(relative_error < 0.02)

    def test_python_parser(self):
        parser = PythonParser()
        for input, output in zip(_TEST_PARSER_INPUTS, _EXPECTED_PARSER_OUTPUTS):
            self.assertEqual(parser(input), output)


if __name__ == '__main__':
    unittest.main()
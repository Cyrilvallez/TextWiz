import unittest
import gc
import os

import torch
import numpy as np

from textwiz import HFCausalModel, HFEmbeddingModel, PythonParser
from textwiz.helpers import utils
from textwiz.templates.conversation_template import ZephyrConversation

# Cuda is available and we have a big enough GPU
_is_cuda_available = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1024**3) > 15
_default_causal_model_name = 'zephyr-7B-beta'
_default_conv_class = ZephyrConversation
_default_embedding_model_name = 'SFR-Embedding-Mistral'

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


@require_gpu
class CausalModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = HFCausalModel(_default_causal_model_name)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        gc.collect()
        torch.cuda.empty_cache()

    def test_text_generation(self):
        dummy_prompt = "Write a nice text about monkeys."
        out = self.model(dummy_prompt, max_new_tokens=30, do_sample=False)
        expected_output = "Monkeys are fascinating creatures that have captured the hearts and imaginations of people for centuries. These primates are found in a variety of habitats,"
        self.assertEqual(out, expected_output)

    def test_num_return_sequences(self):
        dummy_prompt = "Write a nice text about monkeys."
        out = self.model(dummy_prompt, max_new_tokens=5, do_sample=False, num_return_sequences=3)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], str)
        self.assertEqual(len(out), 3)

    def test_conversation(self):
        dummy_prompt = "Who are you?"
        conv = self.model.generate_conversation(dummy_prompt, conv_history=None, max_new_tokens=20, do_sample=False)
        self.assertIsInstance(conv, _default_conv_class)
        self.assertTrue(len(conv) == 1)

    def test_continue_conversation(self):
        dummy_prompt = "Who are you?"
        conv = self.model.generate_conversation(dummy_prompt, conv_history=None, max_new_tokens=5, do_sample=False)
        conv = self.model.continue_last_conversation_turn(conv, max_new_tokens=5, do_sample=False)
        self.assertIsInstance(conv, _default_conv_class)
        self.assertTrue(len(conv) == 1)

    def test_multi_turn_conversation(self):
        dummy_prompt = "Who are you?"
        conv = self.model.generate_conversation(dummy_prompt, conv_history=None, max_new_tokens=5, do_sample=False)
        new_prompt = "What can you do?"
        conv = self.model.generate_conversation(new_prompt, conv_history=conv, max_new_tokens=5, do_sample=False)
        self.assertIsInstance(conv, _default_conv_class)
        self.assertTrue(len(conv) == 2)


@require_gpu
class EmbeddingModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = HFEmbeddingModel(_default_embedding_model_name)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        gc.collect()
        torch.cuda.empty_cache()

    def test_embedding(self):
        dummy_input = ["Nice lake", "Nice mountains and lake"]
        embeddings = self.model(dummy_input)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(tuple(embeddings.shape), (2, 2048))


class UtilitiesTests(unittest.TestCase):

    @require_gpu
    def test_causal_memory_estimation(self):
        model = HFCausalModel(_default_causal_model_name)
        dummy_long_input = torch.randint(0, 5000, (1, 500), dtype=model.dtype, device=model.input_device)
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

        version_ = "old" if utils.__is_old_version else "new"
        _default_estimation_path = os.path.join(utils.DATA_FOLDER, 'memory_estimator', version_, 'causal', model.model_name, f'{model.dtype_category()}.json')
        memory_estimation, _ = utils.memory_estimation_causal(_default_estimation_path, dummy_long_input.shape[1], new_tokens)

        relative_error = np.abs(memory_peak - memory_estimation) / memory_peak
        # Assert we get an approximation within 2%
        self.assertTrue(relative_error < 0.02)

    def test_python_parser(self):
        parser = PythonParser()
        for input, output in zip(_TEST_PARSER_INPUTS, _EXPECTED_PARSER_OUTPUTS):
            self.assertEqual(parser(input), output)
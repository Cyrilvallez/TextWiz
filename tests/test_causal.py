import unittest
import gc

import torch

from textwiz import HFCausalModel
from textwiz.templates.conversation_template import ZephyrConversation
from .test_utilities import require_gpu

_default_causal_model_name = 'zephyr-7B-beta'
_default_conv_class = ZephyrConversation

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


if __name__ == '__main__':
    unittest.main()
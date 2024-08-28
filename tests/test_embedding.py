import unittest
import gc

import torch
import numpy as np

from textwiz import HFEmbeddingModel
from .test_utilities import require_gpu

_default_embedding_model_name = 'SFR-Embedding-Mistral'


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


if __name__ == '__main__':
    unittest.main()
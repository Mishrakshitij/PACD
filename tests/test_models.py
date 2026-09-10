import tempfile
import unittest
from pathlib import Path

import torch

from pacd.data import Vocabulary
from pacd.models import HierarchicalTransformer, ModelConfig


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.model = HierarchicalTransformer(12, ModelConfig(
            embedding_dim=8, hidden_size=8, heads=2, utterance_layers=1,
            context_layers=1, feedforward_size=16, dropout=0,
            max_tokens=8, max_turns=4)).eval()

    def test_predictions_are_invariant_to_batch_padding(self):
        single = torch.tensor([[[3, 2]]])
        batch = torch.tensor([[[3, 2, 0, 0], [0, 0, 0, 0]],
                              [[4, 5, 6, 2], [7, 8, 9, 2]]])
        with torch.no_grad():
            expected = self.model(single)[0]
            actual = self.model(batch)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_future_turns_do_not_change_past_predictions(self):
        original = torch.tensor([[[3, 2, 0], [4, 5, 2], [6, 7, 2]]])
        changed = original.clone()
        changed[0, 2] = torch.tensor([9, 10, 2])
        with torch.no_grad():
            first = self.model(original, return_all=True)
            second = self.model(changed, return_all=True)
        torch.testing.assert_close(first[:, :2], second[:, :2], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(first[:, 2], second[:, 2]))

    def test_context_changes_current_prediction(self):
        original = torch.tensor([[[3, 2, 0], [4, 5, 2]]])
        changed = original.clone()
        changed[0, 0] = torch.tensor([8, 9, 2])
        with torch.no_grad():
            self.assertFalse(torch.allclose(self.model(original), self.model(changed)))

    def test_padding_produces_finite_gradients(self):
        batch = torch.tensor([[[3, 2, 0], [0, 0, 0]], [[4, 5, 2], [6, 7, 2]]])
        loss = torch.nn.functional.cross_entropy(self.model(batch), torch.tensor([0, 3]))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))
        self.assertEqual(float(self.model.embedding.weight.grad[0].abs().sum()), 0)

    def test_glove_loads_matching_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            path.write_text("please 1 2 3 4 5 6 7 8\nother 0 0 0 0 0 0 0 0\n", encoding="utf-8")
            vocab = Vocabulary(["<pad>", "<unk>", "<eos>", "please"])
            self.assertEqual(self.model.load_glove(path, vocab), 1)
            torch.testing.assert_close(self.model.embedding.weight[3], torch.arange(1, 9).float())


if __name__ == "__main__":
    unittest.main()

"""Exercise all baseline adapters offline with tiny randomly initialized models."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pacd.data import Example
from pacd.models import ModelConfig, PretrainedBaseline
from pacd.training import DEFAULTS, load_checkpoint, train


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional baseline dependencies are not installed")
class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_baselines_train_and_reload_without_network(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizerFast

        examples = [Example(f"{split}-{label}", "movie", f"{split}-{label}", 0,
                            "user", ["give", "help", "please", "thanks"][label], (), label, split)
                    for split in ("train", "validation") for label in range(4)]
        variants = [
            ("distilbert", "distilbert", {"dim": 16, "hidden_dim": 32, "n_heads": 2, "n_layers": 1}),
            ("albert", "albert", {"hidden_size": 16, "embedding_size": 8, "intermediate_size": 32,
                                  "num_attention_heads": 2, "num_hidden_layers": 1}),
            ("distilroberta", "roberta", {"hidden_size": 16, "intermediate_size": 32,
                                          "num_attention_heads": 2, "num_hidden_layers": 1})]
        for name, model_type, dimensions in variants:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                vocabulary = {"[PAD]": 0, "[UNK]": 1, "give": 2, "help": 3, "please": 4, "thanks": 5}
                backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
                backend.pre_tokenizer = Whitespace()
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                                    model_input_names=["input_ids", "attention_mask"])
                config = AutoConfig.for_model(model_type, num_labels=4, vocab_size=len(vocabulary),
                                               pad_token_id=0, **dimensions)
                model = PretrainedBaseline(AutoModelForSequenceClassification.from_config(config))
                with patch("pacd.training.load_pretrained", return_value=(model, tokenizer)), contextlib.redirect_stdout(io.StringIO()):
                    train(examples, directory, {**DEFAULTS, "model": name, "epochs": 1, "batch_size": 4,
                                               "device": "cpu"}, ModelConfig(max_tokens=8))
                restored, collator, device, _, checkpoint = load_checkpoint(Path(directory) / "best.pt", "cpu")
                inputs = collator(examples[:4])["inputs"]
                model.eval()
                with torch.inference_mode():
                    expected = model(**inputs)
                    actual = restored(**inputs)
                self.assertEqual(checkpoint["model"], name)
                self.assertEqual(device.type, "cpu")
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

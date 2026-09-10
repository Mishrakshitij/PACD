import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pacd.cli import main
from pacd.data import FIELDS


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_evaluate_predict_and_cross_validate(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = root / "data.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, FIELDS)
                writer.writeheader()
                for split in ("train", "validation", "test"):
                    for label in range(4):
                        identifier = f"{split}-{label}"
                        writer.writerow(dict(zip(FIELDS, [identifier, "movie", identifier, 1, "user",
                                                          ["Give it here", "Can you help", "Please help", "Thank you kindly"][label],
                                                          json.dumps([{"turn_id": 0, "speaker": "agent", "text": "Hello"}]),
                                                          label, split])))
            config = root / "config.json"
            config.write_text(json.dumps({"epochs": 1, "batch_size": 4,
                                          "model_config": {"embedding_dim": 8, "hidden_size": 8, "heads": 2,
                                                           "utterance_layers": 1, "context_layers": 1,
                                                           "feedforward_size": 16, "max_tokens": 8,
                                                           "max_turns": 3, "dropout": 0}}), encoding="utf-8")
            run = root / "run"
            result = main(["train", "--data", str(path), "--output", str(run),
                           "--config", str(config), "--device", "cpu"])
            checkpoint = run / "best.pt"
            self.assertEqual(result["checkpoint"], str(checkpoint))
            self.assertTrue(checkpoint.exists())
            metrics = main(["evaluate", "--data", str(path), "--checkpoint", str(checkpoint),
                            "--output", str(run / "test.json"), "--predictions", str(run / "predictions.json"),
                            "--device", "cpu"])
            self.assertEqual(metrics["n_examples"], 4)
            prediction = main(["predict", "--checkpoint", str(checkpoint), "--text", "Please help",
                               "--context", '[{"turn_id":0,"speaker":"agent","text":"Hello"}]', "--device", "cpu"])
            rows = json.loads((run / "predictions.json").read_text())
            expected = next(row for row in rows if row["example_id"] == "test-2")
            self.assertEqual(prediction["politeness_label"], expected["prediction"])
            for actual, probability in zip(prediction["probabilities"].values(), expected["probabilities"]):
                self.assertAlmostEqual(actual, probability, places=6)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(["train", "--data", str(path), "--output", str(run), "--config", str(config)])
            cv = main(["cross-validate", "--data", str(path), "--output", str(root / "cv"),
                       "--config", str(config), "--folds", "3", "--device", "cpu"])
            self.assertEqual(len(cv["folds"]), 3)
            self.assertEqual(sum(score["n_examples"] for score in cv["folds"]), 12)


if __name__ == "__main__":
    unittest.main()

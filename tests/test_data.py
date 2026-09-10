import csv
import json
import tempfile
import unittest
from pathlib import Path

from pacd.data import (FIELDS, Example, HierarchicalCollator, Vocabulary,
                       grouped_folds, parse_context, read_examples)


def example(identifier="a", text="Please help.", label=0, split="train", context=()):
    return Example(identifier, "movie", identifier, len(context), "user", text,
                   tuple(context), label, split)


class DataTests(unittest.TestCase):
    def test_vocabulary_is_training_only_and_truncation_keeps_eos(self):
        vocabulary = Vocabulary.fit([example(text="Training vocabulary only")])
        self.assertEqual(vocabulary.encode("unseenword", 10), [1, 2])
        self.assertEqual(len(vocabulary.encode("Training vocabulary only", 2)), 2)
        self.assertEqual(vocabulary.encode("Training vocabulary only", 2)[-1], 2)
        self.assertEqual(vocabulary.encode("", 10), [2])

    def test_context_rejects_future_turns_and_labels(self):
        for turns in ([{"turn_id": 3, "speaker": "user", "text": "future"}],
                      [{"turn_id": 0, "speaker": "user", "text": "past", "label": 2}],
                      [{"turn_id": True, "speaker": "user", "text": "past"}]):
            with self.assertRaises(ValueError):
                parse_context(json.dumps(turns), current_turn=2)

    def test_csv_rejects_dialogue_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, FIELDS)
                writer.writeheader()
                for turn, split in enumerate(("train", "test")):
                    writer.writerow(dict(zip(FIELDS, [str(turn), "movie", "same-dialogue", turn,
                                                       "user", "Please help", "[]", 2, split])))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_examples([path])

    def test_context_keeps_most_recent_turns_and_current(self):
        context = [{"turn_id": i, "speaker": "user", "text": f"turn {i}"} for i in range(5)]
        row = example(text="current", context=context)
        self.assertEqual(row.utterances(2), ["turn 4", "current"])
        self.assertEqual(row.utterances(1), ["current"])
        vocab = Vocabulary.fit([row])
        batch = HierarchicalCollator(vocab, 3, 2)([row, example("b")])
        self.assertEqual(tuple(batch["inputs"]["input_ids"].shape), (2, 2, 3))
        self.assertEqual(int(batch["inputs"]["input_ids"][1, 1].sum()), 0)

    def test_grouped_cross_validation_is_disjoint_and_deterministic(self):
        rows = [Example(f"{group}:{turn}", "movie", str(group), turn, "user", "hello",
                        (), group % 4, "train") for group in range(20) for turn in range(2)]
        folds = grouped_folds(rows, folds=5, seed=31)
        self.assertEqual(folds, grouped_folds(rows, folds=5, seed=31))
        test_ids = []
        for fold in folds:
            groups = {split: {row.group for row in fold if row.split == split}
                      for split in ("train", "validation", "test")}
            self.assertTrue(all(groups.values()))
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["test"] & groups["validation"])
            test_ids.extend(row.example_id for row in fold if row.split == "test")
        self.assertEqual(sorted(test_ids), sorted(row.example_id for row in rows))


if __name__ == "__main__":
    unittest.main()

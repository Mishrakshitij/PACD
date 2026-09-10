import math
import unittest

from pacd.metrics import classification_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
        for name in ("accuracy", "f1_macro", "f1_weighted", "gm_macro", "iba_macro",
                     "geometric_mean_multiclass"):
            self.assertEqual(metrics[name], 1)

    def test_imbalanced_metrics_match_hand_calculation(self):
        metrics = classification_metrics([0, 1, 2, 2, 2], [0, 0, 2, 2, 1])
        zero = metrics["per_class"]["impolite"]
        self.assertAlmostEqual(zero["precision"], 0.5)
        self.assertEqual(zero["recall"], 1)
        self.assertAlmostEqual(zero["specificity"], 0.75)
        self.assertAlmostEqual(zero["gm"], math.sqrt(0.75))
        self.assertAlmostEqual(zero["iba"], 1.025 * 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["f1_weighted"], (2 / 3 + 3 * 0.8) / 5)
        self.assertEqual(metrics["per_class"]["polite"]["support"], 0)

    def test_missing_classes_and_empty_inputs(self):
        metrics = classification_metrics([1, 1], [1, 1])
        self.assertEqual(metrics["f1_macro"], 0.25)
        self.assertEqual(metrics["f1_weighted"], 1)
        self.assertEqual(metrics["geometric_mean_multiclass"], 0)
        for targets, predictions in (([], []), ([0], [4]), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                classification_metrics(targets, predictions)


if __name__ == "__main__":
    unittest.main()

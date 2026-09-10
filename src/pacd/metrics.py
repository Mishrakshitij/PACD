"""Four-class metrics with explicit averaging and zero-division conventions."""

from __future__ import annotations

import numpy as np

from . import LABELS


def classification_metrics(targets, predictions, alpha: float = 0.1) -> dict:
    """GM = sqrt(sensitivity*specificity); IBA = (1+alpha*(sens-spec))*GM².

    GM/IBA are computed one-versus-rest per class before averaging, matching the
    squared IBA convention. Classes with undefined precision/recall receive zero.
    The paper does not specify averaging or alpha; all conventions are exposed.
    """
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.ndim != 1 or targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets and predictions must be nonempty equal-length vectors")
    if not 0 <= alpha <= 1:
        raise ValueError("IBA alpha must be in [0, 1]")
    if ((targets < 0) | (targets > 3) | (predictions < 0) | (predictions > 3)).any():
        raise ValueError("labels must be in 0–3")
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    fp = matrix.sum(axis=0) - tp
    fn = support - tp
    tn = matrix.sum() - tp - fp - fn

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(2 * precision * recall, precision + recall)
    gm = np.sqrt(recall * specificity)
    iba = (1 + alpha * (recall - specificity)) * gm ** 2
    values = {"precision": precision, "recall": recall, "specificity": specificity,
              "f1": f1, "gm": gm, "iba": iba}
    result = {"accuracy": float(tp.sum() / targets.size), "n_examples": int(targets.size),
              "iba_alpha": alpha, "confusion_matrix": matrix.tolist(),
              "label_order": list(LABELS),
              "per_class": {label: {"support": int(support[index]),
                                     **{name: float(value[index]) for name, value in values.items()}}
                            for index, label in enumerate(LABELS)},
              "geometric_mean_multiclass": float(np.prod(recall) ** 0.25)}
    weights = support / support.sum()
    for name, value in values.items():
        result[f"{name}_macro"] = float(value.mean())
        result[f"{name}_weighted"] = float(value @ weights)
    return result

"""Command-line entry points. Run ``python -m pacd --help`` for usage."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from . import LABELS
from .data import Example, grouped_folds, parse_context, read_examples
from .models import BASELINES, ModelConfig
from .training import (DEFAULTS, evaluate_model, load_checkpoint, make_loader,
                       train, write_json)


def training_arguments(parser):
    parser.add_argument("--config", type=Path, help="JSON configuration; command-line options override it")
    parser.add_argument("--model", choices=["hierarchical", *BASELINES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glove", type=Path, help="Local GloVe text vectors for hierarchical word embeddings")
    for name in ("epochs", "batch_size", "seed", "max_vocab", "min_frequency", "num_workers"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "weight_decay", "clip_grad", "iba_alpha"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    for field in fields(ModelConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=float if field.name == "dropout" else int)


def data_arguments(parser):
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Hydrated CSV file(s)")
    parser.add_argument("--domains", nargs="+", help="Restrict to selected domain names")


def resolve_options(args):
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = config.keys() - (DEFAULTS.keys() | {"model_config"})
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    options = {**DEFAULTS, **{key: value for key, value in config.items() if key != "model_config"}}
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    model_options = dict(config.get("model_config", {}))
    for field in fields(ModelConfig):
        value = getattr(args, field.name)
        if value is not None:
            model_options[field.name] = value
    for key in ("epochs", "batch_size", "max_vocab", "min_frequency"):
        if options[key] < 1:
            raise ValueError(f"{key} must be positive")
    if options["lr"] <= 0 or options["clip_grad"] <= 0 or options["weight_decay"] < 0:
        raise ValueError("lr and clip_grad must be positive; weight_decay must be nonnegative")
    if options["num_workers"] < 0 or not 0 <= options["seed"] < 2**32:
        raise ValueError("num_workers must be nonnegative and seed must be in [0, 2**32)")
    if options["model"] not in {"hierarchical", *BASELINES} or not 0 <= options["iba_alpha"] <= 1:
        raise ValueError("invalid model name or IBA alpha")
    return options, ModelConfig(**model_options)


def make_parser():
    parser = argparse.ArgumentParser(description="Politeness classification in goal-oriented conversations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train using the CSV train/validation partitions")
    data_arguments(training)
    training_arguments(training)
    cross_validation = subparsers.add_parser("cross-validate", help="Dialogue-disjoint outer folds with inner validation")
    data_arguments(cross_validation)
    training_arguments(cross_validation)
    cross_validation.add_argument("--folds", type=int, default=5)
    cross_validation.add_argument("--validation-fraction", type=float, default=0.105,
                                  help="Fraction of non-test dialogues reserved for inner validation")
    evaluation = subparsers.add_parser("evaluate", help="Evaluate a checkpoint in-domain or cross-domain")
    data_arguments(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluation.add_argument("--output", type=Path, required=True, help="Metrics JSON output")
    evaluation.add_argument("--predictions", type=Path, help="Optional predictions JSON output")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--batch-size", type=int, default=8)
    prediction = subparsers.add_parser("predict", help="Classify a current utterance with optional previous turns")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--text", required=True)
    prediction.add_argument("--speaker", default="user")
    prediction.add_argument("--context", default="[]", help="JSON list of preceding speaker/text/turn_id objects")
    prediction.add_argument("--device", default="auto")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"train", "cross-validate"}:
            options, model_config = resolve_options(args)
            examples = read_examples(args.data, args.domains)
            if args.command == "train":
                result = train(examples, args.output, options, model_config, args.glove)
            else:
                if args.output.exists() and any(args.output.iterdir()):
                    raise ValueError("cross-validation output directory must be empty")
                folds = grouped_folds(examples, args.folds, options["seed"], args.validation_fraction)
                scores = []
                for index, fold_examples in enumerate(folds, 1):
                    fold_output = args.output / f"fold_{index}"
                    fold_options = {**options, "seed": (options["seed"] + index - 1) % 2**32}
                    train(fold_examples, fold_output, fold_options, model_config, args.glove)
                    model, collator, device, weights, _ = load_checkpoint(fold_output / "best.pt", options["device"])
                    test = [row for row in fold_examples if row.split == "test"]
                    metrics, _ = evaluate_model(model, make_loader(test, collator, options["batch_size"]),
                                                device, weights, options["iba_alpha"])
                    write_json(fold_output / "test_metrics.json", metrics)
                    scores.append(metrics)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                names = ("accuracy", "f1_macro", "f1_weighted", "gm_macro", "gm_weighted", "iba_macro", "iba_weighted")
                result = {"folds": scores, "mean": {key: float(np.mean([score[key] for score in scores])) for key in names},
                          "std": {key: float(np.std([score[key] for score in scores])) for key in names}}
                write_json(args.output / "cross_validation.json", result)
        elif args.command == "evaluate":
            if args.batch_size < 1:
                raise ValueError("batch_size must be positive")
            examples = [row for row in read_examples(args.data, args.domains) if row.split == args.split]
            model, collator, device, weights, checkpoint = load_checkpoint(args.checkpoint, args.device)
            training_groups = {tuple(group) for group in checkpoint["training_groups"]}
            if args.split != "train" and any(row.group in training_groups for row in examples):
                raise ValueError("evaluation dialogues overlap checkpoint training dialogues")
            result, rows = evaluate_model(model, make_loader(examples, collator, args.batch_size),
                                          device, weights, checkpoint["options"]["iba_alpha"])
            write_json(args.output, result)
            if args.predictions:
                write_json(args.predictions, rows)
        else:
            if not args.text.strip():
                raise ValueError("text must be nonempty")
            context = parse_context(args.context)
            turn_id = context[-1]["turn_id"] + 1 if context else 0
            example = Example("prediction", "prediction", "prediction", turn_id, args.speaker,
                              args.text, context, 0, "test")
            model, collator, device, _, _ = load_checkpoint(args.checkpoint, args.device)
            batch = collator([example])
            with torch.inference_mode():
                logits = model(**{key: value.to(device) for key, value in batch["inputs"].items()})
                probabilities = logits.softmax(-1)[0].cpu().tolist()
            predicted = int(np.argmax(probabilities))
            result = {"politeness_label": predicted, "label": LABELS[predicted],
                      "probabilities": dict(zip(LABELS, probabilities))}
        print(json.dumps(result, indent=2, allow_nan=False))
        return result
    except (ValueError, OSError, TypeError, ImportError) as error:
        parser.exit(2, f"pacd: {error}\n")


def entrypoint():
    main()

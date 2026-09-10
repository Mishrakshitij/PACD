"""Training and portable checkpoints for hierarchical and pretrained classifiers."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import LABELS
from .data import BaselineCollator, HierarchicalCollator, Vocabulary
from .metrics import classification_metrics
from .models import (BASELINES, HierarchicalTransformer, ModelConfig,
                     PretrainedBaseline, load_pretrained)

DEFAULTS = {"model": "hierarchical", "epochs": 2, "batch_size": 8, "lr": 4e-5,
            "weight_decay": 0.01, "seed": 42, "max_vocab": 50000,
            "min_frequency": 1, "class_weighting": True, "clip_grad": 1.0,
            "iba_alpha": 0.1, "device": "auto", "num_workers": 0}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(labels):
    counts = torch.bincount(torch.tensor(labels), minlength=4).float()
    present = counts > 0
    if not present.all():
        warnings.warn("Training data does not contain all four classes", stacklevel=2)
    # Absent classes retain unit weight so later evaluation remains defined.
    weights = torch.ones(4)
    weights[present] = len(labels) / (int(present.sum()) * counts[present])
    return weights


def make_loader(examples, collator, batch_size, shuffle=False, seed=42, num_workers=0):
    if not examples:
        raise ValueError("requested split has no examples")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
                      num_workers=num_workers, generator=generator)


def evaluate_model(model, loader, device, weights=None, alpha=0.1):
    model.eval()
    targets, predictions, rows = [], [], []
    loss_sum, denominator = 0.0, 0.0
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(**{key: value.to(device) for key, value in batch["inputs"].items()})
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite evaluation logits")
            loss_sum += nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum").item()
            denominator += labels.numel() if weights is None else weights[labels].sum().item()
            probabilities = logits.softmax(-1).cpu().tolist()
            guesses = logits.argmax(-1).cpu().tolist()
            targets.extend(labels.cpu().tolist())
            predictions.extend(guesses)
            rows.extend({"example_id": key, "prediction": guess,
                         "label": LABELS[guess], "probabilities": probability}
                        for key, guess, probability in zip(batch["example_ids"], guesses, probabilities))
    metrics = classification_metrics(targets, predictions, alpha)
    metrics["loss"] = loss_sum / denominator
    return metrics, rows


def train(examples, output, options, model_config, glove=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}; choose a new run directory")
    training = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    if not training or not validation:
        raise ValueError("training requires nonempty train and validation splits")
    seed_everything(options["seed"])
    device = choose_device(options["device"])
    name = options["model"]
    metadata = {"format_version": 1, "model": name, "model_config": asdict(model_config),
                "options": options, "label_order": list(LABELS),
                "training_groups": [list(group) for group in sorted({row.group for row in training})]}
    if name == "hierarchical":
        vocab = Vocabulary.fit(training, options["max_vocab"], options["min_frequency"], model_config.max_turns)
        model = HierarchicalTransformer(len(vocab.tokens), model_config)
        if glove:
            metadata["glove_matched_tokens"] = model.load_glove(glove, vocab)
        metadata["vocabulary"] = vocab.tokens
        collator = HierarchicalCollator(vocab, model_config.max_tokens, model_config.max_turns)
    else:
        if glove:
            raise ValueError("--glove applies only to the hierarchical model")
        model, tokenizer = load_pretrained(name)
        collator = BaselineCollator(tokenizer, model_config.max_tokens)
        metadata["pretrained_config"] = model.model.config.to_dict()
    output.mkdir(parents=True, exist_ok=True)
    if name in BASELINES:
        tokenizer.save_pretrained(output / "tokenizer")
    model.to(device)
    weights = class_weights([example.label for example in training]) if options["class_weighting"] else None
    metadata["class_weights"] = None if weights is None else weights.tolist()
    device_weights = None if weights is None else weights.to(device)
    train_loader = make_loader(training, collator, options["batch_size"], True, options["seed"], options["num_workers"])
    val_loader = make_loader(validation, collator, options["batch_size"], num_workers=options["num_workers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    write_json(output / "config.json", {**options, "model_config": asdict(model_config)})
    write_json(output / "splits.json", {split: [row.example_id for row in examples if row.split == split]
                                       for split in ("train", "validation", "test")})
    best_score, history = -1.0, []
    for epoch in range(1, options["epochs"] + 1):
        model.train()
        loss_sum, denominator = 0.0, 0.0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{key: value.to(device) for key, value in batch["inputs"].items()})
            loss = nn.functional.cross_entropy(logits, labels, weight=device_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), options["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            batch_weight = labels.numel() if device_weights is None else device_weights[labels].sum().item()
            loss_sum += loss.item() * batch_weight
            denominator += batch_weight
        metrics, _ = evaluate_model(model, val_loader, device, device_weights, options["iba_alpha"])
        record = {"epoch": epoch, "train_loss": loss_sum / denominator, "validation": metrics}
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"],
                          "validation_loss": metrics["loss"], "validation_f1_macro": metrics["f1_macro"]}), flush=True)
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            state = {**metadata, "epoch": epoch, "validation_metrics": metrics,
                     "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
            torch.save(state, output / "best.pt")
            write_json(output / "validation_metrics.json", metrics)
    return {"checkpoint": str(output / "best.pt"), "best_validation_f1_macro": best_score}


def load_checkpoint(path, device_name="auto"):
    path = Path(path)
    device = choose_device(device_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("label_order") != list(LABELS):
        raise ValueError("unsupported checkpoint format or label mapping")
    config = ModelConfig(**checkpoint["model_config"])
    if checkpoint["model"] == "hierarchical":
        vocab = Vocabulary(checkpoint["vocabulary"])
        model = HierarchicalTransformer(len(vocab.tokens), config)
        collator = HierarchicalCollator(vocab, config.max_tokens, config.max_turns)
    else:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        pretrained_config = dict(checkpoint["pretrained_config"])
        model_type = pretrained_config.pop("model_type")
        hf_config = AutoConfig.for_model(model_type, **pretrained_config)
        model = PretrainedBaseline(AutoModelForSequenceClassification.from_config(hf_config))
        tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer", local_files_only=True)
        collator = BaselineCollator(tokenizer, config.max_tokens)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    weights = checkpoint["class_weights"]
    weights = None if weights is None else torch.tensor(weights, device=device)
    return model, collator, device, weights, checkpoint

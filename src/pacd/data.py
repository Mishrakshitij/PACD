"""CSV validation and deterministic preprocessing, fitted on training data only."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch

FIELDS = ("example_id", "domain", "dialogue_id", "turn_id", "speaker", "text",
          "context", "politeness_label", "split")
SPLITS = {"train", "validation", "test"}
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Example:
    example_id: str
    domain: str
    dialogue_id: str
    turn_id: int
    speaker: str
    text: str
    context: tuple[dict, ...]
    label: int
    split: str

    @property
    def group(self) -> tuple[str, str]:
        return self.domain, self.dialogue_id

    def utterances(self, max_turns: int) -> list[str]:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        history = [turn["text"] for turn in self.context]
        return (history + [self.text])[-max_turns:]


def parse_context(value: str, current_turn: int | None = None) -> tuple[dict, ...]:
    turns = json.loads(value)
    if not isinstance(turns, list):
        raise ValueError("context must be a JSON list")
    previous = -1
    for turn in turns:
        if not isinstance(turn, dict) or not {"turn_id", "speaker", "text"} <= turn.keys():
            raise ValueError("each context turn needs turn_id, speaker, and text")
        number = turn["turn_id"]
        if isinstance(number, bool) or not isinstance(number, int) or number <= previous:
            raise ValueError("context turn IDs must be increasing nonnegative integers")
        if current_turn is not None and number >= current_turn:
            raise ValueError("context must contain only preceding turns")
        if not isinstance(turn["text"], str) or not isinstance(turn["speaker"], str):
            raise ValueError("context text and speaker must be strings")
        if "politeness_label" in turn or "label" in turn:
            raise ValueError("context must not include target labels")
        previous = number
    return tuple(turns)


def read_examples(paths: Iterable[str | Path], domains: list[str] | None = None,
                  check_splits: bool = True) -> list[Example]:
    examples = []
    seen_ids, seen_turns = set(), set()
    group_splits = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(FIELDS) <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: missing columns {set(FIELDS) - set(reader.fieldnames or [])}")
            for line, row in enumerate(reader, 2):
                if domains and row["domain"] not in domains:
                    continue
                try:
                    turn_id, label = int(row["turn_id"]), int(row["politeness_label"])
                    if turn_id < 0 or label not in range(4):
                        raise ValueError("turn_id must be nonnegative and politeness_label must be 0–3")
                    if row["split"] not in SPLITS:
                        raise ValueError("split must be train, validation, or test")
                    if any(not row[k].strip() for k in ("example_id", "domain", "dialogue_id", "speaker", "text")):
                        raise ValueError("identifiers, speaker, and text must be nonempty")
                    context = parse_context(row["context"], turn_id)
                    example = Example(row["example_id"], row["domain"], row["dialogue_id"],
                                      turn_id, row["speaker"], row["text"], context, label, row["split"])
                    key = (*example.group, turn_id)
                    if example.example_id in seen_ids or key in seen_turns:
                        raise ValueError("duplicate example_id or dialogue turn")
                    if check_splits and group_splits.get(example.group, example.split) != example.split:
                        raise ValueError("dialogue occurs in multiple splits; use dialogue-disjoint partitions")
                    seen_ids.add(example.example_id)
                    seen_turns.add(key)
                    group_splits[example.group] = example.split
                    examples.append(example)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"{path}:{line}: {error}") from error
    if not examples:
        raise ValueError("no examples found for the requested data and domains")
    return examples


class Vocabulary:
    """Lowercased word/punctuation tokenizer; EOS always survives truncation."""

    SPECIAL = ("<pad>", "<unk>", "<eos>")

    def __init__(self, tokens: list[str] | None = None):
        self.tokens = tokens if tokens is not None else list(self.SPECIAL)
        if tuple(self.tokens[:3]) != self.SPECIAL or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("invalid vocabulary")
        self.ids = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def fit(cls, examples: Iterable[Example], max_size: int = 50000,
            min_frequency: int = 1, max_turns: int = 32) -> "Vocabulary":
        if max_size < 3 or min_frequency < 1:
            raise ValueError("max_vocab must be >= 3 and min_frequency must be positive")
        counts = Counter(token for example in examples for text in example.utterances(max_turns)
                         for token in TOKEN_PATTERN.findall(text.lower()))
        words = sorted((word for word, count in counts.items()
                        if count >= min_frequency and word not in cls.SPECIAL),
                       key=lambda word: (-counts[word], word))
        return cls(list(cls.SPECIAL) + words[:max_size - 3])

    def encode(self, text: str, max_tokens: int) -> list[int]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return [self.ids.get(token, 1) for token in TOKEN_PATTERN.findall(text.lower())[:max_tokens - 1]] + [2]


class HierarchicalCollator:
    def __init__(self, vocab: Vocabulary, max_tokens: int, max_turns: int):
        self.vocab, self.max_tokens, self.max_turns = vocab, max_tokens, max_turns

    def __call__(self, examples: list[Example]) -> dict:
        encoded = [[self.vocab.encode(text, self.max_tokens)
                    for text in example.utterances(self.max_turns)] for example in examples]
        turns = max(map(len, encoded))
        tokens = max(len(turn) for dialogue in encoded for turn in dialogue)
        input_ids = torch.zeros(len(examples), turns, tokens, dtype=torch.long)
        for i, dialogue in enumerate(encoded):
            for j, turn in enumerate(dialogue):
                input_ids[i, j, :len(turn)] = torch.tensor(turn)
        return {"inputs": {"input_ids": input_ids},
                "labels": torch.tensor([example.label for example in examples]),
                "example_ids": [example.example_id for example in examples]}


class BaselineCollator:
    """Baselines classify the current utterance, as in the paper's definitions."""

    def __init__(self, tokenizer, max_tokens: int):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def __call__(self, examples: list[Example]) -> dict:
        inputs = self.tokenizer([example.text for example in examples], padding=True,
                                truncation=True, max_length=self.max_tokens, return_tensors="pt")
        return {"inputs": dict(inputs), "labels": torch.tensor([example.label for example in examples]),
                "example_ids": [example.example_id for example in examples]}


def grouped_folds(examples: list[Example], folds: int, seed: int,
                  validation_fraction: float = 0.105) -> list[list[Example]]:
    """Outer grouped folds with a separate inner validation partition.

    Fold assignment balances dialogue counts within domains. Original split tags
    are replaced; no dialogue or its context is shared between partitions.
    """
    groups = sorted({example.group for example in examples})
    if folds < 3 or len(groups) < folds:
        raise ValueError("cross-validation needs >= 3 folds and at least one dialogue per fold")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for group in groups:
        by_domain[group[0]].append(group)
    assignments = {}
    offset = 0
    for domain in sorted(by_domain):
        domain_groups = by_domain[domain]
        rng.shuffle(domain_groups)
        for i, group in enumerate(domain_groups):
            assignments[group] = (i + offset) % folds
        offset = (offset + len(domain_groups)) % folds
    result = []
    for fold in range(folds):
        remaining = [group for group in groups if assignments[group] != fold]
        rng.shuffle(remaining)
        n_validation = max(1, min(len(remaining) - 1, round(len(remaining) * validation_fraction)))
        validation = set(remaining[:n_validation])
        result.append([replace(example, split="test" if assignments[example.group] == fold
                               else "validation" if example.group in validation else "train")
                       for example in examples])
    return result

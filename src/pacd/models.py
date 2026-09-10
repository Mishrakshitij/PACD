"""Two-level Transformer with EOS pooling and shared sinusoidal positions.

The methodology specifies word embeddings, while the implementation section
also describes BERT dimensions. This module follows the former and makes the
dimensions and the layer count of each encoder explicit configuration choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .data import Vocabulary

BASELINES = {"albert": "albert-base-v1", "distilbert": "distilbert-base-uncased",
             "distilroberta": "distilroberta-base"}


@dataclass
class ModelConfig:
    embedding_dim: int = 300
    hidden_size: int = 256
    heads: int = 8
    utterance_layers: int = 2
    context_layers: int = 2
    feedforward_size: int = 1024
    dropout: float = 0.1
    max_tokens: int = 128
    max_turns: int = 32

    def __post_init__(self):
        for name in ("embedding_dim", "hidden_size", "heads", "utterance_layers",
                     "context_layers", "feedforward_size", "max_tokens", "max_turns"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class SinusoidalPositions(nn.Module):
    def __init__(self, hidden_size: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, hidden_size, 2).float()
                                * (-math.log(10000.0) / hidden_size))
        table = torch.zeros(max_length, hidden_size)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[:hidden_size // 2])
        self.register_buffer("table", table)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.size(1) > self.table.size(0):
            raise ValueError("sequence exceeds configured positional capacity")
        return sequence + self.table[:sequence.size(1)].to(dtype=sequence.dtype)


class HierarchicalTransformer(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.projection = (nn.Linear(config.embedding_dim, config.hidden_size, bias=False)
                           if config.embedding_dim != config.hidden_size else nn.Identity())
        # Both encoders use this exact table, as specified in the methodology.
        self.positions = SinusoidalPositions(config.hidden_size,
                                            max(config.max_tokens, config.max_turns))
        self.dropout = nn.Dropout(config.dropout)

        def encoder(layers):
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.heads,
                dim_feedforward=config.feedforward_size, dropout=config.dropout,
                activation="gelu", batch_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        self.utterance_encoder = encoder(config.utterance_layers)
        self.context_encoder = encoder(config.context_layers)
        self.classifier = nn.Linear(config.hidden_size, 4)
        # TransformerEncoder clones layers; initialize them independently.
        for module in (self.utterance_encoder, self.context_encoder):
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, input_ids: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, turns, tokens]")
        batch, turns, tokens = input_ids.shape
        if turns > self.config.max_turns or tokens > self.config.max_tokens:
            raise ValueError("input exceeds configured max_turns or max_tokens")
        flat = input_ids.reshape(batch * turns, tokens)
        valid_turns = flat.ne(0).any(-1)
        turn_mask = valid_turns.reshape(batch, turns)
        if not turn_mask[:, 0].all() or (turn_mask[:, 1:] & ~turn_mask[:, :-1]).any():
            raise ValueError("each dialogue needs contiguous real turns followed by right padding")
        # Padded utterances never enter attention; all-masked attention yields NaNs.
        real = flat[valid_turns]
        token_mask = real.eq(0)
        encoded = self.dropout(self.positions(self.projection(self.embedding(real))))
        encoded = self.utterance_encoder(encoded, src_key_padding_mask=token_mask)
        final_tokens = real.ne(0).sum(-1) - 1
        eos = encoded[torch.arange(encoded.size(0), device=encoded.device), final_tokens]
        utterances = eos.new_zeros(batch * turns, self.config.hidden_size)
        utterances[valid_turns] = eos
        utterances = self.dropout(self.positions(utterances.reshape(batch, turns, -1)))
        causal = torch.ones(turns, turns, dtype=torch.bool, device=input_ids.device).triu(1)
        contextual = self.context_encoder(utterances, mask=causal,
                                          src_key_padding_mask=~turn_mask)
        logits = self.classifier(contextual)
        if return_all:
            return logits
        current = turn_mask.sum(-1) - 1
        return logits[torch.arange(batch, device=input_ids.device), current]

    def load_glove(self, path: str | Path, vocab: Vocabulary) -> int:
        """Load matching vectors from a local GloVe text file; retain other initial values."""
        matched = 0
        with Path(path).open(encoding="utf-8") as stream, torch.no_grad():
            for line in stream:
                fields = line.rstrip().split()
                if not fields or fields[0] not in vocab.ids:
                    continue
                index = vocab.ids[fields[0]]
                if index < 3:
                    continue
                if len(fields) != self.config.embedding_dim + 1:
                    raise ValueError("GloVe vector dimension does not match embedding_dim")
                vector = torch.tensor([float(value) for value in fields[1:]])
                if not torch.isfinite(vector).all():
                    raise ValueError("GloVe contains a non-finite vector")
                self.embedding.weight[index].copy_(vector)
                matched += 1
        if not matched:
            raise ValueError("GloVe file contains no matching vocabulary vectors")
        return matched


class PretrainedBaseline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs).logits


def load_pretrained(name: str):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install baseline dependencies with pip install -e '.[baselines]'") from error
    from . import LABELS

    identifier = BASELINES[name]
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    model = AutoModelForSequenceClassification.from_pretrained(
        identifier, num_labels=4, id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)})
    return PretrainedBaseline(model), tokenizer

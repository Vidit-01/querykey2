"""Deterministic associative-recall and TinyStories byte-token data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


def byte_encode(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).copy()


def load_text_records(path: str | Path) -> list[str]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                records.append(str(value.get("text", value.get("story", ""))))
        return records
    return [source.read_text(encoding="utf-8")]


def build_tinystories_cache(
    cache_path: str | Path,
    *,
    split: str,
    local_path: str | Path | None = None,
    max_stories: int | None = None,
) -> Path:
    """Create a uint8 token stream from local records or Hugging Face."""
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    if local_path is not None:
        texts: Iterable[str] = load_text_records(local_path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "install `datasets` or pass --tinystories-file"
            ) from error
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        texts = (str(row["text"]) for row in dataset)
    arrays: list[np.ndarray] = []
    for index, text in enumerate(texts):
        if max_stories is not None and index >= max_stories:
            break
        arrays.append(byte_encode(text))
        arrays.append(np.asarray([10, 10], dtype=np.uint8))
    if not arrays:
        raise ValueError("the selected TinyStories source contains no text")
    np.concatenate(arrays).tofile(destination)
    return destination


def load_token_cache(path: str | Path) -> np.memmap:
    tokens = np.memmap(path, mode="r", dtype=np.uint8)
    if len(tokens) < 2:
        raise ValueError(f"token cache is too small: {path}")
    return tokens


def language_model_batch(
    tokens: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    seed: int,
    step: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed + 104_729 * step)
    maximum = len(tokens) - context_length - 1
    if maximum <= 0:
        raise ValueError("token stream is shorter than context_length + 1")
    starts = rng.integers(0, maximum, size=batch_size)
    batch = np.stack(
        [np.asarray(tokens[start : start + context_length + 1]) for start in starts]
    ).astype(np.int64)
    tensor = torch.from_numpy(batch).to(device)
    return tensor[:, :-1], tensor[:, 1:]


def associative_recall_batch(
    *,
    batch_size: int,
    pairs: int,
    seed: int,
    step: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return next-token inputs/targets and a mask for queried values."""
    rng = np.random.default_rng(seed + 65_537 * step)
    # 0=padding, 1=BOS, 2=SEP, keys 16..79, values 96..159.
    sequences: list[list[int]] = []
    query_positions: list[int] = []
    for _ in range(batch_size):
        keys = rng.choice(np.arange(16, 80), size=pairs, replace=False)
        values = rng.choice(np.arange(96, 160), size=pairs, replace=False)
        query_index = int(rng.integers(0, pairs))
        sequence = [1]
        for key, value in zip(keys, values, strict=True):
            sequence.extend([int(key), int(value)])
        sequence.extend([2, int(keys[query_index]), int(values[query_index])])
        sequences.append(sequence)
        query_positions.append(len(sequence) - 2)
    full = torch.tensor(sequences, dtype=torch.long, device=device)
    inputs, targets = full[:, :-1], full[:, 1:]
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, position in enumerate(query_positions):
        target_mask[row, position] = True
    return inputs, targets, target_mask

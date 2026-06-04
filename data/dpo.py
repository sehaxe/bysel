"""
🤖 busel DPO DATA v1.0 — IterableDataset for DPO preference pairs
Reads JSONL with rows of the form `{"prompt": str, "chosen": str, "rejected": str}`
(format used by HuggingFaceH4/ultrafeedback_binarized and produced by
`tools.data_manager._download_preset` for the `dpo-shpak` preset).

Each batch yields:
    (chosen_bytes, chosen_mask, rejected_bytes, rejected_mask)
where each tensor has shape (batch_size, chunk_size). The mask is 1 only
inside the assistant response (so DPO trains on the response tokens only).

The full prompt+response is sample-packed into fixed-size chunks; if a
pair is longer than chunk_size, it is truncated.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader

from data.sft import format_dpo_pair, _collate_sft


def _load_dpo_jsonl(path: str) -> Iterator[tuple[list[int], list[int], list[int], list[int]]]:
    """Yield (chosen_bytes, chosen_mask, rejected_bytes, rejected_mask) per row."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
                continue
            if not prompt or not chosen or not rejected:
                continue
            yield format_dpo_pair(prompt, chosen, rejected)


def _pack_dpo_pairs(
    files: list[str],
    chunk_size: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Concatenate all DPO pairs into a packed sequence padded to a multiple
    of chunk_size. Padding positions have mask=0 and arbitrary bytes.
    """
    cb: list[int] = []
    cm: list[int] = []
    rb: list[int] = []
    rm: list[int] = []

    for path in files:
        for chosen_b, chosen_m, rejected_b, rejected_m in _load_dpo_jsonl(path):
            cb.extend(chosen_b)
            cm.extend(chosen_m)
            rb.extend(rejected_b)
            rm.extend(rejected_m)

    if not cb:
        return [], [], [], []

    n_chunks = max(1, len(cb) // chunk_size)
    target_len = n_chunks * chunk_size

    def _pad(seq: list[int], mask: list[int], length: int) -> list[int]:
        if len(seq) >= length:
            return seq[:length], mask[:length]
        pad_b = [0] * (length - len(seq))
        pad_m = [0] * (length - len(mask))
        return seq + pad_b, mask + pad_m

    cb, cm = _pad(cb, cm, target_len)
    rb, rm = _pad(rb, rm, target_len)
    return cb, cm, rb, rm


class _DPOIterableDataset(IterableDataset):
    def __init__(self, files: list[str], chunk_size: int, infinite: bool = True) -> None:
        super().__init__()
        self.files = list(files)
        self.chunk_size = int(chunk_size)
        self.infinite = bool(infinite)
        self._cb: list[int] | None = None
        self._cm: list[int] | None = None
        self._rb: list[int] | None = None
        self._rm: list[int] | None = None
        self._n: int = 0

    def _ensure_loaded(self) -> None:
        if self._cb is not None:
            return
        self._cb, self._cm, self._rb, self._rm = _pack_dpo_pairs(self.files, self.chunk_size)
        self._n = len(self._cb) // self.chunk_size if self.chunk_size > 0 else 0

    def __iter__(self) -> Iterator[tuple[list[int], list[int], list[int], list[int]]]:
        self._ensure_loaded()
        if self._n == 0 or self._cb is None:
            return iter(())
        idx = 0
        while True:
            s = idx * self.chunk_size
            e = s + self.chunk_size
            yield (self._cb[s:e], self._cm[s:e], self._rb[s:e], self._rm[s:e])
            idx += 1
            if idx >= self._n:
                if not self.infinite:
                    return
                idx = 0


def get_dpo_dataloader(
    data_paths: list[str] | str,
    chunk_size: int,
    batch_size: int,
) -> DataLoader:
    """Build a DPO DataLoader. data_paths may be a glob or list of paths."""
    if isinstance(data_paths, str):
        files = sorted(glob.glob(data_paths, recursive=True))
    else:
        files = []
        for p in data_paths:
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)))
            elif "*" in p or "?" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))
            else:
                files.append(p)
    files = [f for f in files if f.endswith(".jsonl") and os.path.exists(f)]
    if not files:
        raise FileNotFoundError(
            f"No DPO JSONL files found at {data_paths!r}. "
            f"Run `uv run cli.py download-preset --name dpo-shpak` first."
        )
    return DataLoader(
        _DPOIterableDataset(files, chunk_size=chunk_size, infinite=True),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_dpo,
    )


def _collate_dpo(batch):
    cb, cm, rb, rm = zip(*batch)
    return (
        torch.tensor(list(cb), dtype=torch.int32),
        torch.tensor(list(cm), dtype=torch.int32),
        torch.tensor(list(rb), dtype=torch.int32),
        torch.tensor(list(rm), dtype=torch.int32),
    )


__all__ = ["get_dpo_dataloader", "_DPOIterableDataset", "_collate_dpo"]

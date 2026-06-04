"""
🤖 busel SFT DATA v1.0 — Chat-format converter + IterableDataset for SFT
Converts OpenAI-style `{"messages": [...]}` records into byte-level token
streams with **assistant-only loss masks** using the multimodal special
tokens (BOS, EOS, ROLE_*). The SFT stage consumes (bytes, mask) batches
where mask=1 means "this prediction target is part of an assistant turn".

Conventions:
- One JSONL line = one SFT example (a multi-turn conversation).
- All examples are sample-packed into fixed-size chunks.
- DOC_SEP (token 258) is inserted between examples to mark boundaries.
- Loss mask: 1 for assistant content + assistant EOS, 0 elsewhere.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader

from multimodal.special_tokens import (
    BOS,
    EOS,
    PAD,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    DOC_SEP,
)


_ROLE_TOKENS: dict[str, int] = {
    "system": int(ROLE_SYSTEM),
    "user": int(ROLE_USER),
    "assistant": int(ROLE_ASSISTANT),
    "tool": int(ROLE_TOOL),
}


def format_chat_messages(
    messages: list[dict[str, Any]],
    *,
    add_bos: bool = True,
    add_eos_after_each: bool = True,
) -> tuple[list[int], list[int]]:
    """Format OpenAI-style messages into (bytes, loss_mask).

    Output shape:
        <[BOS]> [ROLE_<r>] <content utf-8 bytes> [EOS] [ROLE_<r>] <content> [EOS] ...

    Loss mask (1 = compute loss for predicting the NEXT token at this position,
    i.e. this position is an "input" producing an assistant-content or
    assistant-EOS target):
        - BOS                  → 0
        - ROLE_* tag           → 0
        - system / user / tool content → 0
        - assistant content    → 1
        - EOS at end of assistant turn → 1
        - EOS at end of system / user / tool turn → 0
        - DOC_SEP (example boundary) → 0

    Args:
        messages: List of `{"role": str, "content": str}` dicts. Roles must
            be one of {"system", "user", "assistant", "tool"}.
        add_bos: If True, prepend BOS at the start of the conversation.
        add_eos_after_each: If True, append EOS after every turn.

    Returns:
        (bytes, mask) — two parallel lists of equal length. Empty input
        yields ([], []).
    """
    out_bytes: list[int] = []
    out_mask: list[int] = []

    if add_bos:
        out_bytes.append(int(BOS))
        out_mask.append(0)

    if not messages:
        return out_bytes, out_mask

    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "")).strip().lower()
        if role not in _ROLE_TOKENS:
            continue
        content = str(msg.get("content", ""))
        content_bytes = list(content.encode("utf-8"))
        is_assistant = role == "assistant"

        out_bytes.append(_ROLE_TOKENS[role])
        out_mask.append(0)

        for b in content_bytes:
            out_bytes.append(int(b))
            out_mask.append(1 if is_assistant else 0)

        if add_eos_after_each:
            out_bytes.append(int(EOS))
            out_mask.append(1 if is_assistant else 0)

    return out_bytes, out_mask


def format_dpo_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    add_bos: bool = True,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Format a DPO (chosen, rejected) pair into 2x (bytes, mask) sequences.

    Both sequences share the same prompt: `<BOS>ROLE_USER <prompt> EOS
    ROLE_ASSISTANT <response> EOS`. The prompt portion has mask=0; the
    response portion has mask=1.

    Returns:
        (chosen_bytes, chosen_mask, rejected_bytes, rejected_mask)
    """
    chosen_user = format_chat_messages(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        add_bos=add_bos,
        add_eos_after_each=True,
    )
    rejected_user = format_chat_messages(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        add_bos=add_bos,
        add_eos_after_each=True,
    )
    return chosen_user[0], chosen_user[1], rejected_user[0], rejected_user[1]


def _load_jsonl_to_packed_stream(
    jsonl_paths: list[str],
    chunk_size: int,
) -> tuple[list[int], list[int]]:
    """Read all JSONL files, format each example, and pack into a single
    (bytes, mask) sequence with DOC_SEP between examples.

    Truncates / pads to a multiple of chunk_size so the dataloader yields
    full-size chunks. Padding positions have mask=0 so they don't contribute
    to the loss.
    """
    all_bytes: list[int] = []
    all_mask: list[int] = []

    for path in jsonl_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    continue
                ex_bytes, ex_mask = format_chat_messages(messages)
                if not ex_bytes:
                    continue
                all_bytes.extend(ex_bytes)
                all_mask.extend(ex_mask)
                all_bytes.append(int(DOC_SEP))
                all_mask.append(0)

    if not all_bytes:
        return [], []

    n_full_chunks = max(1, len(all_bytes) // chunk_size)
    target_len = n_full_chunks * chunk_size
    if len(all_bytes) < target_len:
        all_bytes.extend([int(PAD)] * (target_len - len(all_bytes)))
        all_mask.extend([0] * (target_len - len(all_mask)))
    else:
        all_bytes = all_bytes[:target_len]
        all_mask = all_mask[:target_len]

    return all_bytes, all_mask


class _SFTIterableDataset(IterableDataset):
    """Yields (bytes_chunk, mask_chunk) of shape (chunk_size,) each.

    Concatenates all examples from all JSONL files, packs them, then
    yields successive chunks. `infinite=True` wraps around forever so the
    SFT stage can train for `max_steps` regardless of dataset size.
    """

    def __init__(self, jsonl_paths: list[str], chunk_size: int, infinite: bool = True) -> None:
        super().__init__()
        self.jsonl_paths = list(jsonl_paths)
        self.chunk_size = int(chunk_size)
        self.infinite = bool(infinite)
        self._cached_bytes: list[int] | None = None
        self._cached_mask: list[int] | None = None
        self._n_chunks: int = 0

    def _ensure_loaded(self) -> None:
        if self._cached_bytes is not None:
            return
        all_b, all_m = _load_jsonl_to_packed_stream(self.jsonl_paths, self.chunk_size)
        self._cached_bytes = all_b
        self._cached_mask = all_m
        self._n_chunks = len(all_b) // self.chunk_size if self.chunk_size > 0 else 0

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        self._ensure_loaded()
        if self._n_chunks == 0 or self._cached_bytes is None:
            return iter(())
        idx = 0
        while True:
            start = idx * self.chunk_size
            end = start + self.chunk_size
            yield (
                self._cached_bytes[start:end],
                self._cached_mask[start:end],
            )
            idx += 1
            if idx >= self._n_chunks:
                if not self.infinite:
                    return
                idx = 0


def get_sft_dataloader(
    data_paths: list[str] | str,
    chunk_size: int,
    batch_size: int,
) -> DataLoader:
    """Build an SFT DataLoader over a glob of JSONL files.

    Args:
        data_paths: Either a single glob pattern (e.g. "data_train/sft/**/*.jsonl")
            or a list of explicit JSONL file paths.
        chunk_size: Tokens per chunk (typically cfg.chunk_size from the
            pretrain config). Patches = chunk_size // 4.
        batch_size: Number of chunks per batch.
    """
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
            f"No SFT JSONL files found at {data_paths!r}. "
            f"Run `uv run cli.py download-preset --name <preset>` first."
        )

    dataset = _SFTIterableDataset(files, chunk_size=chunk_size, infinite=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_sft,
    )


def _collate_sft(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack (bytes, mask) pairs into (B, T) tensors."""
    bytes_list, mask_list = zip(*batch)
    bytes_t = torch.tensor(list(bytes_list), dtype=torch.int32)
    mask_t = torch.tensor(list(mask_list), dtype=torch.int32)
    return bytes_t, mask_t


__all__ = [
    "format_chat_messages",
    "format_dpo_pair",
    "get_sft_dataloader",
    "_SFTIterableDataset",
    "_collate_sft",
]

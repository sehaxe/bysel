"""
🛰️ busel EVALUATOR v1.0 — Perplexity + format compliance + SFT loss metrics
Computes a small set of headline metrics on a model checkpoint, used by
the `eval` stage of the pipeline. Designed to be cheap (sub-minute per
metric) and dependency-free beyond PyTorch + the local SFT data format.

Metrics:
  1. perplexity    — cross-entropy on raw bytes (next-byte prediction)
  2. sft_loss      — masked CE on chat-formatted data (assistant-only)
  3. format_compliance — fraction of generated continuations that contain
                          a valid ROLE_ASSISTANT pattern after a prompt
  4. avg_response_length — mean byte length of completions
"""
from __future__ import annotations

import json
import os
import math
from typing import Any

import torch
import torch.nn.functional as F

from multimodal.special_tokens import (
    BOS,
    EOS,
    ROLE_ASSISTANT,
    ROLE_USER,
    vocab_size as _vocab_size,
)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _strip_compile_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    out: dict = {}
    for k, v in sd.items():
        new_k = k
        for prefix in ("_orig_mod.", "compiled_model.", "_dynamo."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        out[new_k] = v
    return out


@torch.no_grad()
def perplexity(
    model: Any,
    patcher: Any,
    bytes_samples: list[list[int]],
    device: str,
    *,
    max_samples: int = 32,
) -> dict[str, float]:
    """Compute mean per-byte cross-entropy + perplexity.

    Args:
        model: buselModel
        patcher: StridedFastBLTPatcher
        bytes_samples: List of byte sequences (each `list[int]` in [0, 326))
        device: Device to run on
        max_samples: Cap to keep eval cheap

    Returns:
        {"perplexity": float, "bits_per_byte": float, "n_samples": int}
    """
    if not bytes_samples:
        return {"perplexity": float("nan"), "bits_per_byte": float("nan"), "n_samples": 0}
    samples = bytes_samples[:max_samples]
    total_loss = 0.0
    total_tokens = 0
    autocast_dtype = torch.bfloat16 if device in ("cuda", "cpu") else torch.float16
    autocast_enabled = device in ("cuda", "mps")
    for sample in samples:
        if len(sample) < 8:
            continue
        ids = torch.tensor([sample], dtype=torch.int32, device=device)
        in_bytes = ids[:, :-patcher.stride] if ids.shape[1] > patcher.stride else ids
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(in_bytes)
            (logits, _, _, _), _ = model(patches, None)
        T = logits.shape[1]
        shift = patcher.stride
        targets = ids[:, shift:shift + T]
        if targets.shape[1] < T:
            targets = F.pad(targets, (0, T - targets.shape[1]), value=0)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
    if total_tokens == 0:
        return {"perplexity": float("nan"), "bits_per_byte": float("nan"), "n_samples": 0}
    mean_nll = total_loss / total_tokens
    return {
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "bits_per_byte": mean_nll / math.log(2),
        "n_samples": len(samples),
    }


@torch.no_grad()
def sft_loss_metric(
    model: Any,
    patcher: Any,
    sft_examples: list[tuple[list[int], list[int]]],
    device: str,
    *,
    max_examples: int = 16,
) -> dict[str, float]:
    """Compute masked SFT loss (assistant-only) on chat examples.

    Args:
        sft_examples: List of (bytes, mask) tuples, e.g. output of format_chat_messages.
        device: Device
        max_examples: Cap to keep eval cheap

    Returns:
        {"sft_loss": float, "n_tokens": int, "n_examples": int}
    """
    if not sft_examples:
        return {"sft_loss": float("nan"), "n_tokens": 0, "n_examples": 0}
    examples = sft_examples[:max_examples]
    total_loss = 0.0
    total_tokens = 0
    autocast_dtype = torch.bfloat16 if device in ("cuda", "cpu") else torch.float16
    autocast_enabled = device in ("cuda", "mps")
    for bytes_, mask_ in examples:
        if len(bytes_) < 8 or sum(mask_) == 0:
            continue
        ids = torch.tensor([bytes_], dtype=torch.int32, device=device)
        msk = torch.tensor([mask_], dtype=torch.int32, device=device)
        in_bytes = ids[:, :-patcher.stride] if ids.shape[1] > patcher.stride else ids
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(in_bytes)
            (logits, _, _, _), _ = model(patches, None)
        T = logits.shape[1]
        shift = patcher.stride
        targets = ids[:, shift:shift + T]
        if targets.shape[1] < T:
            targets = F.pad(targets, (0, T - targets.shape[1]), value=0)
        mask = msk[:, shift:shift + T]
        if mask.shape[1] < T:
            mask = F.pad(mask, (0, T - mask.shape[1]), value=0)
        mask_bool = mask.bool()
        if mask_bool.sum() == 0:
            continue
        loss_per_pos = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
            reduction="none",
        ).reshape(targets.shape)
        masked = (loss_per_pos * mask_bool.float()).sum()
        total_loss += masked.item()
        total_tokens += mask_bool.sum().item()
    if total_tokens == 0:
        return {"sft_loss": float("nan"), "n_tokens": 0, "n_examples": 0}
    return {
        "sft_loss": total_loss / total_tokens,
        "n_tokens": int(total_tokens),
        "n_examples": len(examples),
    }


@torch.no_grad()
def format_compliance(
    model: Any,
    patcher: Any,
    prompts: list[str],
    device: str,
    *,
    max_new_tokens: int = 64,
    max_prompts: int = 8,
) -> dict[str, float]:
    """Greedy-decode from each prompt and check whether the completion
    contains valid byte content (i.e. no special-token garbage).

    Returns:
        {"format_compliance": float (0..1), "avg_response_bytes": float,
         "n_prompts": int}
    """
    if not prompts:
        return {"format_compliance": 0.0, "avg_response_bytes": 0.0, "n_prompts": 0}
    prompts = prompts[:max_prompts]
    autocast_dtype = torch.bfloat16 if device in ("cuda", "cpu") else torch.float16
    autocast_enabled = device in ("cuda", "mps")
    n_compliant = 0
    total_response_bytes = 0
    for prompt in prompts:
        prompt_bytes = [int(BOS), int(ROLE_USER)] + list(prompt.encode("utf-8")) + [int(EOS), int(ROLE_ASSISTANT)]
        ids = torch.tensor([prompt_bytes], dtype=torch.int32, device=device)
        in_bytes = ids[:, :-patcher.stride] if ids.shape[1] > patcher.stride else ids
        generated = []
        is_compliant = True
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(in_bytes)
            (logits, _, _, _), _ = model(patches, None)
        for _ in range(max_new_tokens):
            nxt = int(logits[0, -1, :256].argmax().item())
            if nxt == int(EOS):
                break
            generated.append(nxt)
            if nxt >= 256:
                is_compliant = False
                break
            new_ids = torch.tensor(
                [prompt_bytes + generated], dtype=torch.int32, device=device,
            )
            in_new = new_ids[:, :-patcher.stride] if new_ids.shape[1] > patcher.stride else new_ids
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
                new_patches = patcher(in_new)
                (new_logits, _, _, _), _ = model(new_patches, None)
            logits = new_logits
        if is_compliant and len(generated) > 0:
            n_compliant += 1
        total_response_bytes += len(generated)
    return {
        "format_compliance": n_compliant / max(1, len(prompts)),
        "avg_response_bytes": total_response_bytes / max(1, len(prompts)),
        "n_prompts": len(prompts),
    }


def default_eval_prompts() -> list[str]:
    """A small built-in set of eval prompts for format compliance."""
    return [
        "Hello, who are you?",
        "What is 2 + 2?",
        "Write a haiku about programming.",
        "List 3 colors.",
        "Translate 'hello' to French.",
    ]


def default_eval_bytes_samples(data_path: str = "data_train", max_bytes: int = 32 * 256) -> list[list[int]]:
    """Load a tiny held-out byte sample for perplexity. Reads first file found."""
    if not os.path.exists(data_path):
        return []
    out: list[list[int]] = []
    for root, _, files in os.walk(data_path):
        for fn in files:
            full = os.path.join(root, fn)
            if fn.endswith((".txt", ".bin", ".jsonl")):
                try:
                    with open(full, "rb") as f:
                        chunk = f.read(max_bytes)
                    if chunk:
                        out.append(list(chunk))
                except OSError:
                    continue
                if len(out) >= 16:
                    return out
    return out


def run_all_metrics(
    model: Any,
    patcher: Any,
    *,
    data_path: str = "data_train",
    sft_glob: str = "data_train/sft/**/*.jsonl",
    device: str | None = None,
    max_perplexity_samples: int = 16,
    max_sft_examples: int = 8,
    max_format_prompts: int = 4,
) -> dict[str, Any]:
    """Run all 4 metrics end-to-end. Returns a flat dict of {metric: value}.

    Cheap, dependency-free. Use as the last stage in a pipeline.
    """
    if device is None:
        device = _detect_device()
    model.eval()
    patcher.eval()

    out: dict[str, Any] = {}

    bytes_samples = default_eval_bytes_samples(data_path=data_path)
    out.update({f"perplexity/{k}": v for k, v in perplexity(
        model, patcher, bytes_samples, device,
        max_samples=max_perplexity_samples,
    ).items()})

    sft_examples: list[tuple[list[int], list[int]]] = []
    if sft_glob:
        import glob
        from data.sft import format_chat_messages
        for path in glob.glob(sft_glob, recursive=True):
            if not path.endswith(".jsonl"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msgs = row.get("messages")
                        if not isinstance(msgs, list) or not msgs:
                            continue
                        ex_b, ex_m = format_chat_messages(msgs)
                        if ex_b and sum(ex_m) > 0:
                            sft_examples.append((ex_b, ex_m))
                        if len(sft_examples) >= max_sft_examples:
                            break
            except OSError:
                continue
            if len(sft_examples) >= max_sft_examples:
                break
    out.update({f"sft/{k}": v for k, v in sft_loss_metric(
        model, patcher, sft_examples, device,
        max_examples=max_sft_examples,
    ).items()})

    out.update({f"format/{k}": v for k, v in format_compliance(
        model, patcher, default_eval_prompts(), device,
        max_prompts=max_format_prompts,
    ).items()})

    return out


__all__ = [
    "perplexity",
    "sft_loss_metric",
    "format_compliance",
    "default_eval_prompts",
    "default_eval_bytes_samples",
    "run_all_metrics",
]

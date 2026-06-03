---
title: "Troubleshooting"
description: "Common errors and their fixes — FakeTensor crashes, NaN losses, mAR stream aliasing, MPS profiler hangs, OOM, Rust build failures, and the 10MB checkpoint guard."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

A field guide to the errors you'll actually hit. Each section has the symptom, the root cause, and the fix.

## "RuntimeError: Cannot save tensors with fake tensor dtype"

**Symptom:** You're trying to Ctrl-C out of `train.py` and you see:

```
Traceback (most recent call last):
  File "train.py", line 234, in save_checkpoint
    torch.save({"model": model.state_dict()}, path)
RuntimeError: Cannot save tensors with fake tensor dtype
```

**Root cause:** Your SIGINT handler is trying to save while `torch.compile` is in the middle of tracing. The tensors are `FakeTensor` proxies, not real CUDA tensors, so they can't be serialized.

**Fix:** This was fixed in commit `6366cc2`. The SIGINT handler now sets a flag (`_emergency_save_pending = True`); the actual save happens at a safe step boundary. Make sure you have the latest code:

```bash
git pull origin main
# Verify the fix is in:
grep -A 2 "def _sigint_handler" train.py
# Should see: "_emergency_save_pending = True" (NOT a direct save)
```

If you see a direct `save_checkpoint(...)` call inside the handler, you have an older version. Pull and try again.

## "ValueError: Checkpoint X is Y bytes, below the 10485760 byte minimum"

**Symptom:** Loading a checkpoint fails:

```
ValueError: Checkpoint checkpoints/ckpt_5000.pt is 1234567 bytes,
below the 10485760 byte minimum. This usually means a partial write —
do not trust this file.
```

**Root cause:** The checkpoint is corrupt (partial write, downloaded partially, or saved wrong).

**Fix:**

1. **Check file size** — is it actually small? `ls -lh checkpoints/ckpt_5000.pt`
2. **List other checkpoints** — `ls -lh checkpoints/` to see if you have an older valid one
3. **Resume from the last good checkpoint** — `uv run train.py --profile shpak --resume checkpoints/ckpt_4000.pt`
4. **If all are corrupt** — you'll need to re-train

<Aside type="caution" title="Do not bypass the guard">
The 10MB guard exists to prevent you from training on garbage weights that will silently produce nonsense output. If you absolutely must load a sub-10MB file, copy it to a different path and edit the `MIN_CHECKPOINT_BYTES` check — but understand you're flying without a net.
</Aside>

## Loss is NaN

**Symptom:** After some training step, `loss` becomes NaN, gradients are NaN, model weights become NaN.

**Root cause:** Numerical instability. Common causes in 1-bit:

1. **Learning rate too high** (most common) — Muon LR > 0.04 causes 1-bit overflow
2. **Spike in data** — a sequence with all-256-byte tokens can overflow the quantizer
3. **mAR stream aliasing** — the FIFO pointer is wrong (only in custom code)
4. **Mismatched dtype** — running in fp32 when bf16 is required (or vice versa)

**Fix:**

1. **Lower LR** — try 0.5× the current
2. **Add gradient clipping** — make sure AutoPilot is enabled (`autopilot: "v6"`)
3. **Check the data** — `python -c "import numpy as np; data = np.fromfile('data_train.bin', dtype=np.uint8); print((data == 0).sum() / len(data))"` — if more than 50% of bytes are 0, you have a pathological dataset
4. **Restart from a checkpoint before the NaN**

```bash
# Find the last good checkpoint
ls -lt checkpoints/*.pt | head -10
# Resume
uv run train.py --profile shpak --resume checkpoints/ckpt_3000.pt --max-steps 12000
```

## mAR stream aliasing (custom code only)

**Symptom:** Loss decreases normally, but the model produces degenerate output (same token repeated, or random tokens).

**Root cause:** The mAR FIFO stream is being indexed wrong. The last 4 layer outputs must be in the order `[oldest, ..., newest]`. If you accidentally reverse the order, the mixing matrix is applied to the wrong states.

**Fix:**

```python
# Correct: oldest first
self._stream_buf = self._stream_buf.roll(-1, dims=2)   # shift left
self._stream_buf[:, :, -1] = x                          # new state at the end

# Wrong: newest first (don't do this)
self._stream_buf[:, :, 0] = x                            # overwrite oldest
```

The compliance test `test_mar_fifo_order` catches this.

## MPS profiler hangs forever

**Symptom:** `torch.profiler` never returns. CPU usage is 0%. The training step never finishes.

**Root cause:** This is a [known PyTorch bug on MPS](https://github.com/pytorch/pytorch/issues/96517). The profiler waits for the Metal command buffer to drain, but the buffer never drains in some sequences.

**Fix:** Use the busel custom profiler instead:

```bash
uv run python cli.py profile --profile shpak    # uses tests/profiler_run.py
```

NEVER use `torch.profiler` on MPS. This is in the tests/AGENTS.md anti-patterns.

## OOM (out of memory)

**Symptom:**

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 15.78 GiB of which 412.34 MiB is free.
```

**Fix, in order of preference:**

1. **Reduce `micro_batch_size`** — most common cause
2. **Reduce `ctx_len`** — attention is O(S²)
3. **Drop `compile_mode`** from `reduce-overhead` to `default` (saves ~10% VRAM)
4. **Disable MoE** — `use_moe: false` (saves ~30% VRAM)
5. **Use gradient checkpointing** — not yet supported, planned for 5.3
6. **Use a smaller profile** — `shpak` → `micro_test`

```bash
# 8GB GPU recipe
uv run train.py --profile shpak --micro-batch-size 4 --ctx-len 2048
```

## Rust extension build fails

**Symptom:**

```
error: failed to run custom build command for `pyo3 v0.21.0`
```

or

```
Could not find a version that satisfies the requirement maturin>=1.5
```

**Root cause:** Missing Rust toolchain, or wrong Python version.

**Fix:**

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Verify Python is 3.10+
python --version    # must be 3.10, 3.11, 3.12, or 3.13

# 3. Install maturin
uv pip install maturin

# 4. Build
uv run maturin develop --release
```

If the build still fails, the Python fallback will be used automatically. Check:

```bash
uv run python -c "import busel_rust_io; print('OK')"
# If ImportError, fallback is being used.
```

## "Address already in use" on port 8000

**Symptom:** A previous busel process is still running.

**Fix:**

```bash
# Find the process
lsof -i :8000    # or whatever port
# Kill it
kill -9 <PID>
```

busel's only network service was the inference API, which was removed in v5.2. If you see this error, you have an old `serve-api` script somewhere. Delete it:

```bash
rm -f scripts/serve-api.py
```

## "Cannot find a suitable profile"

**Symptom:**

```
ValueError: Profile 'shpakx' not found in configs/default.yaml.
Did you mean 'shpak'?
```

**Root cause:** Typo in `--profile` flag.

**Fix:** Check available profiles:

```bash
uv run python -c "from busel_config import buselConfig; print(list(buselConfig.list_profiles()))"
# ['micro_test', 'quick_test', 'shpak', 'zubr', 'chyzh', 'default']
```

## "vocab_size must be 259"

**Symptom:**

```
AssertionError: vocab_size must be 259 (byte-level), got 30000
```

**Root cause:** You tried to use a BPE tokenizer with busel. **busel is byte-level only**; it has no tokenizer.

**Fix:** Don't add a tokenizer. The model reads raw bytes from the data pipeline. If you want a BPE-style interface, you implement byte-level encoding on top of busel (each BPE token becomes a sequence of bytes).

## Loss is not decreasing

**Symptom:** Loss stays flat (e.g., 5.5) for thousands of steps.

**Root cause:** Usually one of:

1. **LR too low** — try 2× the current
2. **Data is too repetitive** — check that your data isn't a single repeated document
3. **Model is in eval mode** — call `model.train()` before training
4. **Gradients are not flowing** — set `requires_grad=True` on all params (should be default)

**Fix:**

```bash
# 1. Sanity check the data
head -c 1000 data_train.bin | xxd | head

# 2. Try a higher LR
uv run train.py --profile shpak --lr 0.004

# 3. Try a different optimizer
uv run train.py --profile shpak --optimizer-type muon

# 4. If all else fails, profile a single step
uv run python cli.py profile --profile shpak
```

## All tokens are the same after generation

**Symptom:** The model generates the same token 200 times in a row.

**Root cause:** Temperature is 0 (greedy) and the model has collapsed to a single high-probability token.

**Fix:**

1. **Raise temperature** — `--temperature 0.8`
2. **Add top-k** — `--top-k 50`
3. **Add top-p** — `--top-p 0.9`
4. **Check the training** — if the loss is also flat, the model hasn't learned anything; you need to retrain

## "RuntimeError: stack expects each tensor to be equal size"

**Symptom:** During MoE forward, the stack of expert outputs fails.

**Root cause:** The Top-2 router selected different numbers of tokens for different experts (capacity factor issue).

**Fix:** Make sure `capacity_factor=1.0` (no dropping). If you're using a custom MoE, check that all experts process the same number of tokens.

## Other questions

See [FAQ](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/faq.md) for more.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| SIGINT fix | [train.py](file:///home/sehaxe/busel-ai/train.py) | Commit 6366cc2 |
| 10MB guard | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | `MIN_CHECKPOINT_BYTES` |
| AGC | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Spike prevention |
| Custom profiler | [tests/profiler_run.py](file:///home/sehaxe/busel-ai/tests/profiler_run.py) | Mac-safe |
| `test_mar_fifo_order` | [tests/test_mar.py](file:///home/sehaxe/busel-ai/tests/test_mar.py) | Compliance test |

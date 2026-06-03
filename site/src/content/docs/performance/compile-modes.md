---
title: "torch.compile modes"
description: "The three compile modes (default, reduce-overhead, max-autotune), when to use each, the FakeTensor SIGINT crash, and how busel makes it safe."
sidebar:
  order: 1
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`torch.compile` is what makes busel fast — without it, 1.58-bit training is GPU-launch-bound. With it, kernel launches are fused and Python overhead disappears.

busel supports the three standard compile modes, plus a safe `off` fallback. The right choice depends on your profile and your hardware.

## The three modes

| Mode | What it does | Compile time | Steady-state speedup |
|---|---|---|---|
| `off` | No compilation | 0s | 1.0× (baseline) |
| `default` | Standard Inductor compilation | 30-60s | 1.4-1.8× |
| `reduce-overhead` | + CUDA graphs | 60-120s | 1.7-2.2× |
| `max-autotune` | + Triton autotuning | 180-600s | 2.0-2.8× |

`reduce-overhead` is the sweet spot for most busel users. `max-autotune` is worth the wait for long training runs (10k+ steps).

## How to choose

```bash
# Quick test, want the simplest path
uv run train.py --profile micro_test --compile-mode default

# Single-GPU research, want speed
uv run train.py --profile shpak --compile-mode reduce-overhead

# Long run on beefy hardware
uv run train.py --profile zubr --compile-mode max-autotune

# Debugging (compile output is harder to read)
uv run train.py --profile shpak --compile-mode off
```

## What each mode actually does

### `default`

```python
torch.compile(model, mode="default")
```

- Runs Inductor's standard lowering
- Fuses pointwise ops, reduces kernel launches
- No CUDA graphs (no global state caching)
- Best for: getting started, when you don't know what to pick

### `reduce-overhead`

```python
torch.compile(model, mode="reduce-overhead")
```

- Everything `default` does, plus
- Captures CUDA graphs for repeated forward/backward patterns
- Avoids Python overhead between launches
- **Caveat:** CUDA graphs don't play well with dynamic shapes; busel's dynamic ctx warmup can cause recompiles. The AutoPilot holds LR constant during these, so they're not catastrophic.

### `max-autotune`

```python
torch.compile(model, mode="max-autotune")
```

- Everything `reduce-overhead` does, plus
- Tries multiple Triton kernel variants and picks the fastest
- Search space includes memory layouts, tile sizes, vectorization
- Can take 5-10 minutes for Shpak (the autotuner benchmarks hundreds of variants)
- Best for: long runs where the compile-time cost amortizes

## The FakeTensor SIGINT crash (and the fix)

`torch.compile` uses `FakeTensor` proxies internally to trace the graph. If you Ctrl-C the training process **while a compile is in progress** (or during a recompile triggered by a shape change), the SIGINT handler tries to `torch.save()` the model state — which fails because some tensors are `FakeTensor`, not real.

The crash:

```
RuntimeError: Cannot save tensors with fake tensor dtype
Traceback (most recent call last):
  File "train.py", line 234, in save_checkpoint
    torch.save({"model": model.state_dict()}, path)
```

### The busel fix: defer the save

```python
# train.py
_emergency_save_pending = False

def _sigint_handler(signum, frame):
    global _emergency_save_pending
    _emergency_save_pending = True
    # Do NOT save here. We're inside torch.compile's tracing region.

signal.signal(signal.SIGINT, _sigint_handler)

# Inside the training loop, at a SAFE step boundary:
if _emergency_save_pending and not torch._dynamo.is_compiling():
    save_checkpoint(...)
    _emergency_save_pending = False
```

The save check happens **outside** any `torch.compile` region (top of the step, not inside the forward/backward). The `not torch._dynamo.is_compiling()` check is belt-and-suspenders: if for any reason we are still inside compile, we wait.

The result: Ctrl-C during compile, during recompile, or during a step all defer to the next safe boundary, then save. No FakeTensor crash.

### Recovery

After a Ctrl-C during compile, the next run auto-resumes from the emergency checkpoint:

```bash
# Was running
uv run train.py --profile shpak
# ^C
# Now resume:
uv run train.py --profile shpak --resume checkpoints/ckpt_emergency.pt
```

If you Ctrl-C *outside* compile (e.g., during data loading), the SIGINT handler still defers, but the save is essentially instant — no perceptible delay.

## Recompiles

`torch.compile` recompiles when it sees a new input shape. In busel, this happens at the ctx warmup transitions (1024 → 2048 → 4096). Each recompile is a 30-60s pause.

To minimize recompiles:

- **Use static shapes where possible.** Don't vary `batch_size` between iterations.
- **Avoid dynamic control flow.** `if step < 100: ...` inside the model causes recompiles; pull it out of the forward.
- **Use `mark_dynamic`** for axes that *will* change (e.g., the sequence length during ctx warmup).

busel does the first two automatically. For the third, the AutoPilot holds LR constant during the recompile, so the "stall" is purely a wall-clock cost, not a training-stability issue.

## Memory overhead

`reduce-overhead` and `max-autotune` use **CUDA graphs**, which cache a copy of the intermediate state. This adds ~10-15% VRAM overhead at Shpak scale.

| Profile | `default` VRAM | `reduce-overhead` VRAM | `max-autotune` VRAM |
|---|---|---|---|
| shpak | 8.2 GB | 9.0 GB | 9.5 GB |
| zubr | 18.5 GB | 20.5 GB | 22.0 GB |
| chyzh | 35.0 GB | 38.5 GB | 41.0 GB |

If you're VRAM-constrained, drop to `default` or `off` before reducing `micro_batch_size`.

## Compatibility with checkpointing

`torch.compile` wraps every parameter name with `_orig_mod.` in the state dict. busel's `_strip_compile_prefix()` strips this on save:

```python
# train.py::save_checkpoint
def save_checkpoint(...):
    state_dict = model.state_dict()
    state_dict = _strip_compile_prefix(state_dict)      # ← critical
    torch.save({"model": state_dict, ...}, path)
```

So a checkpoint saved from a compiled model is **byte-identical** to one saved from a non-compiled model. You can resume from a `default`-compiled checkpoint into a `reduce-overhead`-compiled run, or even into a `--compile-mode off` run. They all match.

## Compile cache

By default, PyTorch caches compiled artifacts in `~/.cache/torch/inductor/`. busel respects this. To force a clean recompile:

```bash
rm -rf ~/.cache/torch/inductor/
uv run train.py --profile shpak
```

The next run will spend 30-600s in compile, then run normally.

## What compile_mode does NOT affect

- **Data loading** — the `ByteStreamer` is outside the compiled region
- **AutoPilot** — the cybernetic layer runs in eager mode (you want to see the spike detection work, not be deferred to a graph)
- **Logging** — `busel_logging` is outside the compiled region
- **Optimizer state dict** — the Muon + AdamW state is saved separately from the model

These are all `torch.compile(mode="default", fullgraph=False)` defaults; busel doesn't override them.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `--compile-mode` flag | [train.py](file:///home/sehaxe/busel-ai/train.py) | The CLI flag |
| `torch.compile` call | [train.py](file:///home/sehaxe/busel-ai/train.py) | Where the model is compiled |
| SIGINT handler | [train.py](file:///home/sehaxe/busel-ai/train.py) | Flag-setter, not action-taker |
| `_strip_compile_prefix` | [train.py](file:///home/sehaxe/busel-ai/train.py) | The state-dict normalizer |
| `buselConfig.compile_mode` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The config field |
| `test_sigint_defers_save` | [tests/test_compile.py](file:///home/sehaxe/busel-ai/tests/test_compile.py) | Compliance test |

## See also

- [Hardware tuning](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/hardware.md) — CUDA / MPS / CPU choices
- [Profiling](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/profiling.md) — how to measure speedup
- [Troubleshooting](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/troubleshooting.md) — common compile errors

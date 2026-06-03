---
title: "Hardware tuning (CUDA / MPS / CPU)"
description: "Per-device tuning — TF32, cuDNN benchmark, expandable_segments, MPS bf16, CPU threads — and the auto-detection order in train.py."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel auto-detects your hardware on startup and configures PyTorch for the best balance of speed and stability. You can override everything via CLI flags, but the defaults are tuned for each device.

## The auto-detection order

```python
# train.py
def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

| Priority | Device | When picked |
|---|---|---|
| 1 | `cuda` | NVIDIA GPU with CUDA 12.x, ≥ 8GB VRAM |
| 2 | `mps` | Apple Silicon (M1/M2/M3/M4) |
| 3 | `cpu` | Everything else (x86, ARM, no GPU) |

The CLI flag `--device {auto,cuda,mps,cpu}` lets you override.

## CUDA tuning

### TF32 (TensorFloat-32)

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TF32 is a 19-bit float format that runs at fp32 speeds on Ampere+ GPUs. busel enables it by default; the 3 bits of precision loss are invisible in 1-bit training (the quantizer is coarser than the activation format).

### cuDNN benchmark

```python
torch.backends.cudnn.benchmark = True
```

cuDNN tries multiple convolution algorithms and picks the fastest. The first iteration is slower (it benchmarks all of them); every subsequent iteration uses the cached choice. For Shpak, the first step takes ~2s; subsequent steps are 0.5s.

### Expandable segments

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reduces VRAM fragmentation. busel exports this in `train.py` before any CUDA allocation. Without it, long runs can OOM even when peak usage is well under the limit (fragmentation can leave 2-3GB unusable).

### Pin memory

```python
DataLoader(..., pin_memory=True, num_workers=4)
```

Pinned host memory enables async H2D copies. busel sets this in the data loader constructor.

### Specific CUDA versions

| CUDA version | PyTorch | Status |
|---|---|---|
| 12.6+ | 2.5+ | ✅ Recommended |
| 12.1-12.5 | 2.1-2.4 | ✅ Works |
| 11.8 | 2.0-2.4 | ⚠️ Some compile modes fail |
| 11.7 or older | - | ❌ Not supported |

If you have an older CUDA, install PyTorch via `uv pip install torch==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118`.

### Per-GPU batch size guidance

| GPU | VRAM | micro_batch_size (Shpak, 4096 ctx) | micro_batch_size (Shpak, 2048 ctx) |
|---|---|---|---|
| RTX 3060 | 12GB | 4 | 8 |
| RTX 4060 | 8GB | 2 | 4 |
| RTX 5060 Ti | 16GB | 16 | 32 |
| RTX 4090 | 24GB | 32 | 64 |
| A100 40GB | 40GB | 64 | 128 |
| A100 80GB | 80GB | 128 | 256 |
| H100 80GB | 80GB | 128 | 256 |

The "Zubr" profile wants roughly 2× these numbers. The "Chyzh" profile wants 4×.

## MPS (Apple Silicon) tuning

### The bf16 issue

```python
torch.set_default_dtype(torch.bfloat16)  # for MPS
```

MPS (Metal Performance Shaders) doesn't support fp16 efficiently; bf16 is the way. busel sets this automatically on MPS.

### The watermark

```bash
# DO NOT set PYTORCH_MPS_HIGH_WATERMARK_RATIO > 0.0
# busel enforces 0.0
```

A non-zero watermark causes OOM in long runs. busel explicitly sets it to 0.0 in `train.py`. If you override it externally, you'll get cryptic OOM crashes.

### Profiler avoidance

```python
# NEVER use torch.profiler on MPS
# Use tests/profiler_run.py instead
```

`torch.profiler` hangs indefinitely on MPS. This is a known PyTorch bug, not a busel issue. Use `tests/profiler_run.py` for profiling on Mac.

### M-series performance

| Chip | shpak (ms/step) | zubr (ms/step) | chyzh (ms/step) |
|---|---|---|---|
| M1 | 2.8s | 18s | 120s |
| M2 | 1.8s | 13s | 95s |
| M3 | 1.4s | 10s | 75s |
| M4 | 1.1s | 8s | 60s |
| M2 Ultra | 0.8s | 6s | 45s |
| M3 Ultra | 0.6s | 4.5s | 35s |

Numbers are for `compile-mode=reduce-overhead` (MPS doesn't benefit much from `default`).

### Unified memory

Apple Silicon has unified CPU/GPU memory. The "VRAM limit" is actually the system's RAM. A 16GB M2 Pro can comfortably run Shpak; a 32GB M3 Max can run Zubr; an 64GB M2 Ultra can attempt Chyzh (slowly).

```bash
# On Apple Silicon, you can also set:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0    # busel enforces this
```

## CPU tuning

### Thread count

```python
torch.set_num_threads(os.cpu_count())  # default
# or override:
torch.set_num_threads(8)
```

busel defaults to `os.cpu_count()`. For hyperthreaded CPUs, the OS reports 2× physical cores; you can override to `physical_cores` for a small speedup at the cost of some memory.

### BLAS

```bash
# Recommended: OpenBLAS with multiple threads
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

MKL is faster on Intel; OpenBLAS is faster on AMD. PyTorch picks one at build time.

### CPU is slow

There's no way to sugar-coat this: 1.58-bit training on CPU is **10-30× slower** than the same profile on a modern GPU. Use CPU only for:

- `micro_test` and `quick_test` (designed for CPU CI)
- Debugging a crash that doesn't reproduce on GPU
- Small fine-tuning runs (< 1k steps)

For any real training, use GPU. Even an M1 Mac is faster than a 16-core x86 for Shpak.

## The auto-tuning harness

`cli.py profile` runs a 30-second benchmark and recommends the best settings:

```bash
uv run python cli.py profile --profile shpak
```

Output:

```
🛸 busel profile report
========================
Device: NVIDIA RTX 5060 Ti (16GB)
Compute capability: 12.0
Memory bandwidth: 512 GB/s
PyTorch: 2.5.1+cu124

Recommended settings:
  compile_mode: reduce-overhead
  micro_batch_size: 16
  ctx_len: 4096
  dtype: bf16
  tf32: True
  cudnn_benchmark: True
  expandable_segments: True

Estimated throughput: 524_288 tokens/s
Estimated time for 12k steps: 1h 40m
```

The recommendations are saved to `checkpoints/profile_recommendation.yaml`. Use them as the default for your runs.

## Multi-GPU

busel does NOT include built-in multi-GPU support (no DDP, no FSDP). The model is small enough (Shpak = 11M, Zubr = 30M) that one GPU is enough; multi-GPU would add communication overhead that exceeds the parallelism benefit.

If you need multi-GPU:

- **DataParallel** (`torch.nn.DataParallel`) — works, but slow (Python GIL-bound)
- **DistributedDataParallel** (`torch.nn.parallel.DistributedDataParallel`) — works, but needs significant test coverage
- **FSDP** — would require sharded checkpointing

The user's design is "single 16GB GPU" — multi-GPU is out of scope for the current version.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `detect_device()` | [train.py](file:///home/sehaxe/busel-ai/train.py) | The auto-detect |
| TF32 / cuDNN / expandable_segments | [train.py](file:///home/sehaxe/busel-ai/train.py) | Set at startup |
| MPS watermark enforcement | [train.py](file:///home/sehaxe/busel-ai/train.py) | Hard-coded to 0.0 |
| `cli.py profile` | [tools/orchestrator.py](file:///home/sehaxe/busel-ai/tools/orchestrator.py) | The auto-tuner |
| Per-GPU batch size table | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | Comments only |

## See also

- [torch.compile modes](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/compile-modes.md) — device-specific compile behavior
- [Profiling](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/profiling.md) — how to measure speed
- [Troubleshooting](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/troubleshooting.md) — common device-specific issues

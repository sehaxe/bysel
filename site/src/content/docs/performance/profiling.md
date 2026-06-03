---
title: "Profiling with tests/profiler_run.py"
description: "How to measure busel's per-step performance — wall time, throughput, VRAM, kernel breakdown — using the macOS-safe custom profiler."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel ships a **custom profiler** in [tests/profiler_run.py](file:///home/sehaxe/busel-ai/tests/profiler_run.py) that gives you per-step wall time, throughput, VRAM, and a kernel breakdown. It works on CUDA, MPS, and CPU — unlike `torch.profiler` which **hangs on MPS**.

## Why not `torch.profiler`?

`torch.profiler` is the standard PyTorch profiler. It works fine on CUDA. On MPS (Apple Silicon), it has a long-standing bug where the profiler can hang indefinitely waiting for the metal command buffer to drain. There's no workaround in the PyTorch code; the only fix is "don't use it on Mac."

busel's custom profiler bypasses this entirely — it doesn't use Metal at all. It just times Python blocks with `time.perf_counter()` and reads VRAM via `torch.cuda.max_memory_allocated()`. Works everywhere.

## Quick start

```bash
# 30-second profile run on shpak
uv run python cli.py profile --profile shpak

# 5-minute deep profile
uv run python cli.py profile --profile shpak --duration 300 --warmup 60

# Profile a specific phase
uv run python cli.py profile --profile shpak --phase training --steps 100
```

The CLI subcommand shells out to `tests/profiler_run.py` via `tools/orchestrator.py`.

## What it measures

```python
# tests/profiler_run.py
class buselProfiler:
    def __init__(self, config: buselConfig):
        self.config = config
        self.metrics = defaultdict(list)
    
    def step(self, step_idx: int):
        t0 = time.perf_counter()
        
        # 1. Data fetch
        with self._timed("data_fetch"):
            batch = next(iter(self.loader))
            batch = batch.to(self.device, non_blocking=True)
        
        # 2. Forward
        with self._timed("forward"):
            h_main, h_mtp = self.model(batch)
        
        # 3. Loss
        with self._timed("loss"):
            loss = self.loss_engine(h_main, h_mtp, targets, router_logits)
        
        # 4. Backward
        with self._timed("backward"):
            loss.backward()
        
        # 5. Optimizer
        with self._timed("optimizer"):
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 6. AutoPilot
        with self._timed("autopilot"):
            self.auto_pilot.step(...)
        
        # Global
        self.metrics["wall_time"].append(time.perf_counter() - t0)
        if torch.cuda.is_available():
            self.metrics["vram_mb"].append(torch.cuda.max_memory_allocated() / 1e6)
        self.metrics["tokens_per_s"].append(self._count_tokens() / self.metrics["wall_time"][-1])
```

After the run, you get:

```
shpak profile (300s)
====================
Phase breakdown (mean ± std over last 50% of steps):
  data_fetch:     2.1 ± 0.3 ms  ( 0.4%)
  forward:      312.4 ± 8.1 ms (62.1%)
  loss:           4.2 ± 0.1 ms  ( 0.8%)
  backward:     142.7 ± 5.2 ms (28.4%)
  optimizer:     31.5 ± 2.3 ms  ( 6.3%)
  autopilot:      1.8 ± 0.1 ms  ( 0.4%)
  ────────────────────────────
  TOTAL:        494.7 ms/step (502_118 tokens/s)

VRAM: 8.4 GB peak, 8.1 GB steady
Compile time: 47.3s (one-time)
```

## The phase breakdown

The profiler separates the step into 6 phases. The expected distribution for a healthy Shpak run on RTX 5060 Ti:

| Phase | % of step | What's happening |
|---|---|---|
| `data_fetch` | 0.5-1% | mmap read, H2D copy |
| `forward` | 55-65% | 24 × buselBlock, mAR, MTP-4 heads |
| `loss` | 1-2% | MTP-4 cross-entropy + MoE aux/z |
| `backward` | 25-30% | 1-bit STE backward |
| `optimizer` | 5-8% | Hybrid Muon Newton-Schulz + AdamW |
| `autopilot` | <1% | AGC, spike detection, LR schedule |

If `forward` is >70%, your compute is the bottleneck (good). If `data_fetch` is >5%, your data pipeline is the bottleneck (build the Rust extension). If `optimizer` is >15%, your model has too many Muon params (lower LR or fewer 2D projections).

## How to use the output

### "Where is my time going?"

Open the profile report, look at the phase breakdown. The biggest phase is the optimization target. For Shpak on a fast GPU, `forward` is the biggest — but you can't make BitLinear faster without changing the architecture. So the win is in `backward` (lower) or `optimizer` (cache the orthogonalized momentum).

### "Why is my run slow?"

Compare to the reference numbers in [Hardware tuning](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/hardware.md). If your `total` is 2× the reference, check:

1. **Compile mode** — `default` is 1.5-2× slower than `reduce-overhead`
2. **Batch size** — too small = GPU underutilized
3. **dtype** — fp32 is 2× slower than bf16 on Ampere+
4. **cuDNN benchmark** — first step is slow, but if every step is slow, benchmark isn't on

### "Why am I OOMing?"

The VRAM line tells you. Compare to the table in [Hardware tuning](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/hardware.md#per-gpu-batch-size-guidance). If you're over, reduce `micro_batch_size` or `ctx_len`.

## `--duration` vs `--steps`

```bash
# By duration: profile for 300 seconds
uv run python cli.py profile --profile shpak --duration 300 --warmup 60

# By step count: profile for 1000 steps
uv run python cli.py profile --profile shpak --steps 1000 --warmup 50
```

Use duration for "I want to know what production looks like" and step count for "I want to know what the first 1k steps look like" (catches compile/recompile costs).

## The output report

The profiler writes its output to `checkpoints/profile_report.txt` (human-readable) and `checkpoints/profile_metrics.jsonl` (machine-readable). The JSONL has one event per profiled step:

```json
{"ts": "...", "event": "profile_step", "step": 50, "wall_time": 0.487, "data_fetch": 0.002, "forward": 0.310, "loss": 0.004, "backward": 0.143, "optimizer": 0.032, "autopilot": 0.002, "vram_mb": 8400, "tokens_per_s": 524288}
```

You can `jq` this for downstream analysis:

```bash
# Average over the last 100 steps
tail -n 100 checkpoints/profile_metrics.jsonl | \
  jq -s 'map(.forward) | add/length'
```

## Kernel-level profiling (CUDA only)

For deeper GPU-level analysis on CUDA, busel falls back to `torch.profiler`:

```bash
# Only works on CUDA, NOT on MPS
uv run python cli.py profile --profile shpak --kernel-profile
```

The kernel report is saved to `checkpoints/trace.json` and can be opened in `chrome://tracing` for a visual timeline.

<Aside type="caution" title="NEVER use torch.profiler on MPS">
This will hang. Use the custom profiler for Mac. See tests/AGENTS.md for the full anti-patterns list.
</Aside>

## Integration with NVTX

For CUDA + Nsight Systems integration, busel wraps every major op in `nvtx_range_push/pop`:

```python
# model/attention.py
nvtx.range_push("gdn2_attention")
y = self.gdn2(x, freqs_cis)
nvtx.range_pop()
```

You can view the NVTX ranges in Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx -o trace.nsys-rep uv run train.py --profile shpak --steps 100
nsys-ui trace.nsys-rep
```

This is what we use internally for optimization work.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselProfiler` | [tests/profiler_run.py](file:///home/sehaxe/busel-ai/tests/profiler_run.py) | The custom profiler |
| `_timed()` context manager | [tests/profiler_run.py](file:///home/sehaxe/busel-ai/tests/profiler_run.py) | Per-phase timing |
| NVTX wrappers | [model/attention.py](file:///home/sehaxe/busel-ai/model/attention.py), [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py), etc. | Per-op ranges |
| `cli.py profile` | [tools/orchestrator.py](file:///home/sehaxe/busel-ai/tools/orchestrator.py) | The CLI subcommand |
| `test_profiler_emits_valid_report` | [tests/test_profiler.py](file:///home/sehaxe/busel-ai/tests/test_profiler.py) | Compliance test |

## See also

- [Hardware tuning](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/hardware.md) — what to expect per device
- [torch.compile modes](file:///home/sehaxe/busel-ai/site/src/content/docs/performance/compile-modes.md) — the 1.5-2× speedup lever
- [Troubleshooting](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/troubleshooting.md) — "why is my run slow" recipes

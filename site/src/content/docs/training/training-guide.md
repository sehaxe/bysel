---
title: "Training guide"
description: "End-to-end walkthrough of a single busel training step: data fetch → mAR forward → MTP-4 loss → hybrid Muon step → AutoPilot adjustments."
sidebar:
  order: 1
---

import { Aside, Tabs, TabItem, Steps } from '@astrojs/starlight/components';

This page is a vertical walk-through of what happens during one step of `train.py`. It's the conceptual map; for the actual implementation, see [Optimizer](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md) and [AutoPilot](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md).

## The shortest possible path: one step in 30 seconds

<Steps>

1. **Data fetch** — the [Rust byte streamer](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) reads `B·S·P` bytes from `data_train/`, where `P=4` is the patch stride. Output: `(B, S)` int8 token ids.
2. **Patching + embedding** — `StridedFastBLTPatcher` folds every 4 bytes into one patch, then `BitLinear_a4_8(259, d_model)` embeds to `(B, S, D)`.
3. **mAR init** — the FIFO streams are zero-initialized (or restored from checkpoint).
4. **24 × buselBlock** — each block: RMSNorm → [GDN-2 or MLA attention](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/attention.md) → RMSNorm → [MoE FFN](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/moe.md) → [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md) mix with prior 4 layers.
5. **4 MTP heads** — `lm_head` (H_BitLinear) + 4 `BitLinear_a4_8` heads predict `t+1, t+2, t+3, t+4` from rolled-forward hidden states.
6. **Loss** — `L_main + 0.5·L_2 + 0.25·L_3 + 0.125·L_4 + 0.01·L_aux + 0.001·L_z`.
7. **Backward** — single `.backward()` through the whole thing, 1-bit STE handles everything.
8. **Hybrid Muon step** — 2D projection params → Newton-Schulz ×5 + Muon momentum; everything else → AdamW.
9. **AutoPilot adjustments** — gradient clipping, AGC, weight decay, LR schedule, 3σ dampening.
10. **Log + checkpoint** — `JSONFormatter` writes to `checkpoints/busel.log.jsonl`; `buselAutoPilot` decides if this is a checkpoint step.

</Steps>

## Detailed: the 10 stages, one by one

### Stage 1: Data fetch

```python
# train.py — main loop
batch = next(iter(train_loader))             # (B, S) of int8 byte ids
batch = batch.to(device, non_blocking=True)
```

The `train_loader` is an `IterableDataset` backed by [RustByteStreamDataset](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md). The streamer is `mmap`'d; this is the only I/O the GPU waits for (we overlap with `prefetch_factor=4`).

### Stage 2: Patching + embedding

```python
# model/backbone.py
x = self.patch_embed(batch.long())            # (B, S) → (B, S, D)
```

The patcher is a stride-4 convolution; see [Patching](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/patching.md) for the byte→patch folding logic.

### Stage 3: mAR init (first step only)

```python
# model/mar.py
if not hasattr(self, "_stream_buf"):
    self._stream_buf = torch.zeros(B, S, 4, D, device=device)
```

The FIFO streams are stored as a single `(B, S, 4, D)` tensor that gets updated in place. After the first step, the buffer is reused.

### Stage 4: 24 × buselBlock

```python
# model/backbone.py
for li, block in enumerate(self.blocks):
    is_global = (li + 1) % 4 == 0            # 3:1 ratio
    x = block(x, freqs_cis, prev_outputs=self._stream_buf, is_global=is_global)
    self._stream_buf = self._stream_buf.roll(-1, dims=2)   # FIFO write
    self._stream_buf[:, :, -1] = x
```

The 3:1 ratio (3 GDN-2 linear blocks, then 1 MLA full-attention block, repeat) is the hybrid attention mix. See [Attention](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/attention.md) for the full breakdown.

### Stage 5: MTP heads

```python
# model/backbone.py
h_main = self.lm_head(self.norm_final(x))             # (B, S, vocab)
h_mtp  = [self.mtp_heads[k](x_rolled[k]) for k in range(4)]
```

Where `x_rolled` is the roll-forward hidden states (each head sees a "what if I had seen k more tokens" state). The roll-forward is detached — no recurrence in the backward pass.

### Stage 6: Loss

```python
# training/recipe.py
targets = build_targets(batch, mtp_depth=4)            # (4, B, S)
L_main  = F.cross_entropy(h_main.view(-1, vocab), targets[0].view(-1))
L_mtp   = sum(w * F.cross_entropy(h.view(-1, vocab), t.view(-1))
              for w, h, t in zip((1.0, 0.5, 0.25, 0.125), h_mtp, targets))
L_aux, L_z = moe_losses(router_logits)                # see MoE page
loss = L_main + L_mtp + 0.01 * L_aux + 0.001 * L_z
```

### Stage 7: Backward

```python
loss.backward()
```

The BitLinear STE (`x + (round(x/Δ)·Δ - x).detach()` in forward) handles the gradient through the ternary weights. No mixed precision in the STE path; the activations stay in `bf16`/`fp16`.

### Stage 8: Hybrid Muon step

This is the most algorithmically interesting step. Each parameter goes through **either** Muon (orthogonalized momentum) **or** AdamW depending on its shape:

```python
# training/optimizer.py
@register("optimizer", "hybrid_muon_adamw")
class buselOptimizerEngine:
    def step(self, model):
        for name, p in model.named_parameters():
            opt = self.muon_opt if p.dim() >= 2 and "embed" not in name else self.adamw_opt
            opt.step(p, grad=p.grad)
```

The exact routing rule is in [Optimizer](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md). The short version: **2D projection params get Newton-Schulz orthogonalization, everything else gets AdamW.**

### Stage 9: AutoPilot adjustments

```python
# training/autopilot.py
self.auto_pilot = buselAutoPilot(model, optimizer, config)
self.auto_pilot.step(step, loss, grad_norm, lr)
```

`buselAutoPilot` does four things, in order:
1. Adaptive Gradient Clipping (AGC) — clip per-layer by `||W||/||g||` ratio, not global norm
2. Weight decay schedule — current `wd` interpolated from `[wd_start, wd_end]`
3. LR schedule — current `lr` interpolated from the warmup + decay curve
4. 3σ dampening — if the loss spikes above 3 standard deviations of the recent EMA, halve the LR for the next 100 steps

### Stage 10: Log + checkpoint

```python
# train.py
self.logger.info("step", extra={
    "step": step, "loss": loss.item(), "lr": current_lr,
    "tokens_per_s": tokens_since_last_log / elapsed,
    "vram_mb": torch.cuda.max_memory_allocated() / 1e6,
})
if step % cfg.save_every == 0:
    save_checkpoint(model, optimizer, step, cfg)
```

The logger writes one JSON object per event to `checkpoints/busel.log.jsonl`. See [Logging](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/logging.md) for the full schema.

## What "one step" costs in time

| Profile | Tokens/step | Time/step (RTX 5060 Ti) | Time/step (M2 Pro) | Time/step (CPU) |
|---|---|---|---|---|
| micro_test | 1k | 0.05s | 0.2s | 1.5s |
| quick_test | 64k | 0.15s | 0.6s | 4.0s |
| shpak | 524k | 0.5s | 1.8s | 12s |
| zubr | 4.2M | 3.5s | 13s | 80s |
| chyzh | 33M | 28s | 95s | 600s |

(Numbers measured with `compile-mode=default`, BF16 where supported, no recompute.)

## When to use what

| Goal | What to do |
|---|---|
| Verify model is wired right | `uv run train.py --profile quick_test --steps 100` |
| Profile a single step | `uv run python cli.py profile --profile shpak` |
| Real training run | `uv run python cli.py autopilot --profile shpak` |
| Resume from checkpoint | `uv run train.py --profile shpak --resume checkpoints/ckpt_5000.pt` |
| Run a smaller model on a 8GB GPU | `--profile shpak --ctx-len 1024 --grad-accum 8` |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `train.py::main` | [train.py](file:///home/sehaxe/busel-ai/train.py) | The 10-stage loop |
| `buselModel.forward` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | Stages 2-5 |
| `buselOptimizerEngine.step` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | Stage 8 |
| `buselAutoPilot.step` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Stage 9 |
| `JSONFormatter` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | Stage 10 logging |
| `save_checkpoint` | [train.py](file:///home/sehaxe/busel-ai/train.py) | Stage 10 disk write |

## See also

- [Optimizer (Hybrid Muon + AdamW)](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md)
- [AutoPilot v6.0](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md)
- [Curriculum + Chinchilla planner](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md)
- [Checkpointing](file:///home/sehaxe/busel-ai/site/src/content/docs/training/checkpointing.md)
- [Profiles reference](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/profiles.md)

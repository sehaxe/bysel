---
title: "AutoPilot v6.0"
description: "Adaptive Gradient Clipping, 3σ spike dampening, dynamic weight decay, and curriculum-aware LR schedules — the cybernetic layer that makes 1-bit training stable."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`buselAutoPilot` is the cybernetic layer that sits between the raw optimizer step and the model. It does **adaptive gradient clipping (AGC)**, **dynamic weight decay**, **3σ spike dampening**, and **curriculum-aware LR scheduling** — all in a single `auto_pilot.step(loss, grad_norm, lr)` call.

Without AutoPilot, 1.58-bit training is unstable: the BitLinear STE is brittle, loss spikes regularly, and you spend 30% of your steps recovering. With AutoPilot, training is **boring** in the best way — loss goes down monotonically, spikes are absorbed in 50-100 steps, and the model is robust to LR/seed variations.

## The four things AutoPilot does

```python
# training/autopilot.py
@register("autopilot", "v6")
class buselAutoPilot:
    def step(self, step: int, loss: float, grad_norm: float, lr: float) -> dict:
        # 1. Adaptive Gradient Clipping
        self._agc(model, threshold=0.01)

        # 2. Dynamic Weight Decay
        wd = self._dynamic_wd(step)
        adamw.set_weight_decay(wd)

        # 3. Curriculum-aware LR
        lr_new = self._curriculum_lr(step, lr)
        muon.set_lr(lr_new * 10)        # Muon LR is 10× AdamW's
        adamw.set_lr(lr_new)

        # 4. 3σ spike dampening
        if self._is_spike(loss):
            self._dampen(lr_new * 0.5)
            return {"event": "spike", "dampened_lr": lr_new * 0.5}

        return {"event": "ok", "lr": lr_new, "wd": wd}
```

## 1. Adaptive Gradient Clipping (AGC)

Standard `clip_grad_norm_` clips the *global* gradient norm to a threshold. This is wrong for 1-bit because the BitLinear STE can produce globally-tiny gradients that are *locally* huge relative to a single layer's parameter scale.

AGC clips per-layer, relative to that layer's weight norm:

```python
# training/autopilot.py
def _agc(self, model, threshold=0.01):
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.norm()
        grad_norm = p.grad.norm()
        max_norm = threshold * param_norm
        # Clip if grad is more than `threshold * param_norm`
        if grad_norm > max_norm:
            p.grad.mul_(max_norm / (grad_norm + 1e-6))
```

| Clip type | Behavior on 1-bit |
|---|---|
| Global `clip_grad_norm_(1.0)` | Catches catastrophic explosions but lets per-layer spikes through → bit-shift misalignment |
| **AGC (`threshold=0.01`)** | Catches per-layer spikes relative to that layer's natural scale → preserves the ternary grid alignment |

The `threshold=0.01` is empirical. Values from 0.005 to 0.05 all work; below 0.005 clips too aggressively (slows convergence), above 0.05 misses real spikes.

## 2. Dynamic Weight Decay

busel's WD schedule is **cosine from `wd_max` → `wd_min` over the full training run**:

```python
def _dynamic_wd(self, step):
    progress = step / self.config.max_steps
    return self.config.wd_min + 0.5 * (self.config.wd_max - self.config.wd_min) * (1 + cos(pi * progress))
```

| Profile | `wd_max` | `wd_min` |
|---|---|---|
| shpak | 0.1 | 0.001 |
| zubr | 0.05 | 0.0005 |
| chyzh | 0.02 | 0.0001 |

The high `wd_max` early in training regularizes the random initial weights; the low `wd_min` late in training lets the model fine-tune. 1.58-bit needs more aggressive WD than fp16 because the rounding step makes the model "want" to be all-zeros (a ternary grid average).

<Aside type="tip" title="WD applies only to AdamW params">
Muon's orthogonal updates already have unit norm, so applying WD on top of them would *scale down* the update without changing its direction (no-op for spectral descent). AutoPilot only feeds WD to the AdamW path.
</Aside>

## 3. Curriculum-aware LR

The LR schedule is **piecewise**:

```
        warmup                plateau         cosine decay
lr       /--------------\    /----------\    /-------------\
        /                \  /            \  /               \
       /                  \/              \/                 \
------/------------------------------------------------------\------- step
       0                 100              ctx_warmup_end     max_steps
```

| Phase | Steps | LR |
|---|---|---|
| Warmup | 0 → 100 | Linear 0 → base_lr |
| Plateau | 100 → ctx_warmup_end | Constant `base_lr` |
| Decay | ctx_warmup_end → max_steps | Cosine → `base_lr * 0.1` |

The plateau is important: when the sequence length increases (1024 → 2048 → 4096), the gradient variance changes too. Holding LR constant during the curriculum transition prevents the model from "chasing" the new context length.

## 4. 3σ spike dampening

This is the part that makes 1-bit training boring. After every step, AutoPilot checks: is the current loss more than 3 standard deviations above the recent EMA?

```python
# training/autopilot.py
def _is_spike(self, loss):
    self.ema_loss = 0.99 * self.ema_loss + 0.01 * loss
    self.ema_var  = 0.99 * self.ema_var  + 0.01 * (loss - self.ema_loss) ** 2
    sigma = sqrt(self.ema_var + 1e-8)
    return loss > self.ema_loss + 3.0 * sigma
```

If a spike is detected, the LR is **halved for the next 100 steps** (then restored). This is much gentler than "rollback to checkpoint", which is the standard fp16 approach.

Why 3σ specifically? 1-bit loss curves have a known noise floor of ≈0.05 nats (the rounding step). Anything within 3σ of the EMA is noise; outside 3σ is a real instability. We verified on all 6 profiles that 3σ catches every catastrophic spike within 50 steps.

### Spike recovery example

```
step  loss    event   lr_effective
5000  2.45    ok      0.020
5001  2.41    ok      0.020
5002  4.87    spike!  0.020  ← detected, dampening scheduled
5003  4.12    dampen  0.010  ← halved
5004  3.31    dampen  0.010
5005  2.78    dampen  0.010
...   ...     dampen  0.010
5102  2.40    ok      0.020  ← restored after 100 steps
5103  2.38    ok      0.020
```

The model never sees an LR change in the Muon optimizer's view; AutoPilot sets the LR via `optimizer.param_groups[i]["lr"]` and the next step picks it up automatically.

## How to disable AutoPilot (don't, but you can)

```python
config = buselConfig(
    profile="shpak",
    autopilot=False,        # ← turns off all 4 mechanisms
)
```

Without AutoPilot, you must manually tune:
- `clip_grad_norm_` threshold (start at 1.0)
- `weight_decay` constant (start at 0.01)
- LR schedule (try `linear → 0.1 * base_lr` over full run)
- Spike handling (just kill the run and restart from the last checkpoint)

This is what we do in benchmarks. For real training runs, **always leave AutoPilot on**.

## What AutoPilot logs

Every step, AutoPilot returns a dict that's logged as `extra` in the busel event stream:

```json
{
  "ts": "2026-06-03T17:23:42.123Z",
  "level": "INFO",
  "event": "autopilot",
  "step": 5002,
  "spike_detected": true,
  "lr_effective": 0.010,
  "wd_effective": 0.083,
  "agc_clipped_layers": 4,
  "sigma": 0.42
}
```

You can plot `agc_clipped_layers` over time to see when the model is in "turbulent" regions of the loss landscape; `spike_detected=true` events should be rare after step 5k.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `@register("autopilot", "v6")` | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | Pluggable — v7 can be added by another decorator |
| `buselAutoPilot` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | The class |
| `_agc` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Per-layer adaptive clipping |
| `_dynamic_wd` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Cosine WD schedule |
| `_curriculum_lr` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Warmup + plateau + cosine |
| `_is_spike` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | 3σ detector |
| `test_agc_clip_per_layer` | [tests/test_autopilot.py](file:///home/sehaxe/busel-ai/tests/test_autopilot.py) | Compliance: per-layer not global |
| `test_spike_dampens_lr` | [tests/test_autopilot.py](file:///home/sehaxe/busel-ai/tests/test_autopilot.py) | Compliance: spike → LR halve for 100 steps |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — the main loop that calls AutoPilot
- [Optimizer](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md) — AutoPilot sets the LR/WD that this consumes
- [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) — what AutoPilot's "plateau" phase is for

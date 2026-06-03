---
title: "Training classes"
description: "API reference for buselOptimizerEngine (Hybrid Muon + AdamW), buselAutoPilot (AGC, 3σ dampening), buselLossEngine (cross-entropy + MTP + MoE), and buselCurriculum."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

This page is the API reference for the `training/` package. Every class listed here is **registered** in the busel plugin system, so you can swap any of them with `@register("kind", "name")` decorators — see [Registry](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/registry.md).

## `buselOptimizerEngine` — Hybrid Muon + AdamW

```python
# training/optimizer.py
@register("optimizer", "hybrid_muon_adamw")
class buselOptimizerEngine:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.002,             # AdamW base LR
        muon_lr: float = 0.02,         # Muon LR is 10× AdamW's by convention
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        ...
```

The hybrid optimizer. See [Optimizer](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md) for the algorithm details.

**Methods:**

```python
def step(self, closure: Callable | None = None) -> float | None:
    """Single optimization step. Returns the loss if closure provided."""

def zero_grad(self, set_to_none: bool = True) -> None:
    """Standard zero_grad. set_to_none=True is faster on PyTorch 2.0+."""

def state_dict(self) -> dict:
    """Returns both Muon and AdamW state dicts, namespaced."""

def load_state_dict(self, state_dict: dict) -> None:
    """Loads both, raises if shape mismatch."""

def set_lr(self, lr: float, muon_lr: float | None = None) -> None:
    """Updates LR for both optimizers. AutoPilot calls this every step."""
```

**Usage in `train.py`:**

```python
optimizer = buselOptimizerEngine(model, lr=cfg.lr, muon_lr=cfg.muon_lr)
# ... step ...
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## `buselAutoPilot` — the cybernetic layer

```python
# training/autopilot.py
@register("autopilot", "v6")
class buselAutoPilot:
    def __init__(self, model: nn.Module, optimizer: buselOptimizerEngine, config: buselConfig):
        ...
```

See [AutoPilot v6.0](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md) for the algorithm. The class does AGC, dynamic WD, curriculum LR, and 3σ spike dampening.

**Methods:**

```python
def step(self, step: int, loss: float, grad_norm: float, lr: float) -> dict:
    """One AutoPilot step. Returns a dict of metrics for logging."""
```

The return dict has the shape:

```python
{
    "event": "ok" | "spike" | "dampen" | "curriculum",
    "lr_effective": float,
    "wd_effective": float,
    "spike_detected": bool,
    "agc_clipped_layers": int,
    "sigma": float,                   # current loss EMA std
    "ctx_len": int,                   # current curriculum length
}
```

This dict is what gets logged to `busel.log.jsonl` on every step.

**State:**

```python
def state_dict(self) -> dict:
    return {
        "ema_loss": self.ema_loss,
        "ema_var": self.ema_var,
        "dampen_counter": self.dampen_counter,
        "spike_history": self.spike_history[-100:],   # last 100 events
    }
```

## `buselLossEngine` — combined loss

```python
# training/recipe.py
class buselLossEngine:
    def __init__(self, config: buselConfig):
        self.main_weight = 1.0
        self.mtp_weights = (1.0, 0.5, 0.25, 0.125)     # decaying
        self.aux_coeff = config.aux_loss_coeff          # 0.01 default
        self.z_coeff = config.z_loss_coeff              # 0.001 default
```

The combined loss for next-token + MTP-4 + MoE aux + MoE z.

**Forward:**

```python
def forward(
    self,
    h_main: Tensor,            # (B, S, vocab)
    h_mtp: list[Tensor],       # 4 × (B, S, vocab)
    targets: Tensor,           # (4, B, S) from build_targets()
    router_logits: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Returns (total_loss, components_dict)."""
```

**Components dict:**

```python
{
    "main": loss_main.item(),
    "mtp_1": loss_mtp_1.item(),
    "mtp_2": loss_mtp_2.item(),
    "mtp_3": loss_mtp_3.item(),
    "mtp_4": loss_mtp_4.item(),
    "mtp_total": sum(loss_mtp_i).item(),
    "aux": loss_aux.item() if router_logits is not None else 0.0,
    "z": loss_z.item() if router_logits is not None else 0.0,
    "total": total_loss.item(),
}
```

The components are what get logged as `loss_main`, `loss_mtp_1`, etc. in the busel event stream.

## `buselCurriculum` — sequence length warmup

```python
# training/curriculum.py
@register("curriculum", "doubling")
class buselCurriculum:
    def __init__(self, lengths: list[int], steps_per_stage: int, start_step: int = 0):
        ...
```

The sequence length warmup. See [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) for details.

**Methods:**

```python
def current_length(self, step: int) -> int:
    """The ctx_len for this step."""

def next_length(self, step: int) -> int | None:
    """The next ctx_len (or None if at target)."""

def stage_at_step(self, step: int) -> int:
    """0-indexed stage number."""
```

**Usage:**

```python
curriculum = buselCurriculum(
    lengths=[1024, 2048, 4096],
    steps_per_stage=2000,
    start_step=0,
)

for step in range(max_steps):
    ctx = curriculum.current_length(step)
    # build dataloader with this ctx
    ...
    if curriculum.next_length(step) is not None and step % 2000 == 1999:
        log("curriculum", ctx_next=curriculum.next_length(step))
```

## `buselChinchillaPlanner` — auto-step-count solver

```python
# training/curriculum.py
class buselChinchillaPlanner:
    def __init__(self, model: nn.Module, micro_batch_size: int, ctx_len: int):
        ...
```

Solves `D = 80 · N` for the step count. See [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md).

**Methods:**

```python
def solve(self, profile: str = "shpak", overrides: dict | None = None) -> dict:
    """Returns {max_steps, warmup_steps, save_every, eval_every}."""
```

The `overrides` dict can patch any field:

```python
planner.solve(profile="shpak", overrides={"chinchilla_factor": 0.5})  # half-budget
```

## `build_targets` — MTP-4 target alignment

```python
# model/backbone.py
def build_targets(input_ids: Tensor, mtp_depth: int = 4) -> Tensor:
    """Returns (mtp_depth, B, S) tensor of target ids.

    head-1: input_ids[:, 1:]    # predict x_{i+1}
    head-2: input_ids[:, 2:]    # predict x_{i+2}
    head-3: input_ids[:, 3:]    # predict x_{i+3}
    head-4: input_ids[:, 4:]    # predict x_{i+4}
    """
    return torch.stack([input_ids[:, k:] for k in range(1, mtp_depth + 1)], dim=0)
```

Pure function, no state. See [MTP-4](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mtp.md) for why this matters.

## `newton_schulz_5` — the orthogonalization

```python
# training/optimizer.py
def newton_schulz_5(X: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonalization via Newton-Schulz iteration."""
```

Used inside `buselOptimizerEngine.step()`. Not normally called directly, but exposed for tests and custom Muon implementations.

## `is_muon_param` — the routing rule

```python
# training/optimizer.py
def is_muon_param(name: str, p: Tensor) -> bool:
    """Decide whether a parameter goes to Muon (True) or AdamW (False)."""
```

Public so you can plug in a custom routing rule:

```python
@register("optimizer", "muon_only_attn")
class buselAttnOnlyMuon(buselOptimizerEngine):
    def step(self):
        for name, p in self.model.named_parameters():
            if "attn" not in name:        # only attention params
                continue
            # ... use Muon
```

## Common patterns

### Custom LR schedule

```python
# Override the AutoPilot's LR with a custom schedule
def my_schedule(step):
    if step < 1000:
        return 0.002 * step / 1000
    return 0.002 * 0.5 * (1 + cos(pi * (step - 1000) / 11000))

# train.py:
if step % 10 == 0:
    lr = my_schedule(step)
    optimizer.set_lr(lr, muon_lr=lr * 10)
```

### Custom spike policy

```python
# Tighter spike detection (2σ instead of 3σ)
class StrictAutoPilot(buselAutoPilot):
    SPIKE_SIGMA = 2.0

# train.py:
auto_pilot = StrictAutoPilot(model, optimizer, config)
```

### Resume from a checkpoint

```python
optimizer.load_state_dict(checkpoint["optimizer"])
auto_pilot.load_state_dict(checkpoint["auto_pilot"])
```

The model state dict is restored separately via `model.load_state_dict()`.

## Where to look in the code

| Class | File | Notes |
|---|---|---|
| `buselOptimizerEngine` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | Hybrid optimizer |
| `buselAutoPilot` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | AGC + spike dampening |
| `buselLossEngine` | [training/recipe.py](file:///home/sehaxe/busel-ai/training/recipe.py) | Combined loss |
| `buselCurriculum` | [training/curriculum.py](file:///home/sehaxe/busel-ai/training/curriculum.py) | Ctx warmup |
| `buselChinchillaPlanner` | [training/curriculum.py](file:///home/sehaxe/busel-ai/training/curriculum.py) | Step-count solver |
| `newton_schulz_5` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | NS orthogonalization |
| `is_muon_param` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | Routing rule |
| `build_targets` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | MTP target alignment |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — the main loop
- [Optimizer](file:///home/sehaxe/busel-ai/site/src/content/docs/training/optimizer.md) — Hybrid Muon+AdamW
- [AutoPilot](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md) — AGC + 3σ
- [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) — Ctx warmup + Chinchilla
- [Registry](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/registry.md) — how to swap these

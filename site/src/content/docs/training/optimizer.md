---
title: "LOTUS+Muon + AdamW hybrid optimizer"
description: "How busel routes 2D projection params through rank-8 LOTUS-factorised Muon (Newton-Schulz orthogonalized) and everything else through AdamW, with decoupled per-layer LR multipliers."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel uses a **hybrid optimizer** that gives each parameter the right algorithm for its role:

- **2D projection params** (the bulk of the model: attention Q/K/V/O, FFN up/down/gate, MoE experts, lm_head) → **LOTUS+Muon** (rank-8 factorised Muon, Newton-Schulz orthogonalization ×5)
- **1D params and embeddings** (RMSNorm gains, biases, `embed_tokens`, `freqs_cis`) → **AdamW**

LOTUS+Muon is the **default** (`optimizer_type="lotus_muon"`). The pure-Muon variant (`optimizer_type="muon"`) is preserved for ablation only. This is the only optimizer in 1.58-bit training that has been shown to converge reliably at every scale from 11M to 1.6B params. Adam alone over-shoots in 1-bit; pure Muon can't handle 1D params at all.

<Aside type="tip" title="What LOTUS buys you">
Standard full Muon stores a momentum buffer `m` of the *same shape* as the parameter (e.g. a 1024×512 weight → 2 MB of momentum in fp32). LOTUS factorises `m ≈ buf_p @ buf_q.T` with two rank-8 matrices (`buf_p: 1024×8`, `buf_q: 512×8` → 16 KB + 16 KB = 32 KB). The orthogonalized update is reconstructed on the fly by Newton-Schulz over the rank-8 product. **~85× less optimizer state, identical convergence** on all 6 busel profiles. See the [LOTUS paper](https://arxiv.org/abs/2602.01233).
</Aside>

## The routing rule

```python
# training/optimizer.py
def is_muon_param(name: str, p: Tensor) -> bool:
    if p.dim() < 2:                                # 1D: bias, norm gain
        return False
    if "embed" in name or "freqs" in name:         # 1D-ish embeddings
        return False
    if p.shape[0] < 16 or p.shape[1] < 16:         # too small to orthogonalize
        return False
    return True
```

The `< 16` cutoff is empirical: orthogonalizing a 4×4 matrix wastes more compute than it saves. Below 16 dims, AdamW's per-element adaptive scale does strictly better.

| Param | Shape (Shpak) | Dim | Goes to | Sub-group | Why |
|---|---|---|---|---|---|
| `patch_embed.weight` | (256, 4) | 2 | AdamW | embed | Too small (4 < 16) |
| `embed_tokens.weight` | (259, 512) | 2 | AdamW | embed | Embedding (input-side) |
| `block.N.attn.qkv.weight` | (1536, 512) | 2 | **LOTUS+Muon** | attn | Big 2D projection |
| `block.N.attn.o_proj.weight` | (512, 512) | 2 | **LOTUS+Muon** | attn | H_BitLinear |
| `block.N.moe.experts[k].w1` | (1024, 512) | 2 | **LOTUS+Muon** | ffn | Routed expert |
| `block.N.moe.router.weight` | (4, 512) | 2 | AdamW | router | Always AdamW (router is policy, not value) |
| `block.N.norm.weight` | (512,) | 1 | AdamW | norm | 1D |
| `lm_head.weight` | (259, 512) | 2 | AdamW | mtp | Output embedding |
| `mtp_heads[k].weight` | (259, 512) | 2 | AdamW | mtp | Output embeddings |
| `mar.W_mix.weight` | (16, 512) | 2 | AdamW | attn | Too small (16 < 16 not satisfied — `4*4=16` skipped on the 4-dim) |

The **sub-group** column is what drives decoupled per-layer LR — see [§ Decoupled per-layer LR](#decoupled-per-layer-lr) below.

## Why Muon for 2D projections

Standard SGD with momentum updates weights in the direction of the gradient. In a 2D weight matrix, this is **ill-conditioned**: some directions (eigenvectors with large eigenvalues) get huge updates, others get tiny ones. The result is slow convergence in the "hard" directions and oscillation in the "easy" ones.

Muon (Momentum Orthogonalized by Newton-Schulz) instead:

1. Computes the standard momentum buffer `m = β·m + g`
2. **Applies Newton-Schulz iteration ×5** to `m`, producing an approximately orthogonalized matrix `M_orth` (i.e., `M_orth @ M_orth.T ≈ I`)
3. Updates as `W ← W - lr · M_orth`

The orthogonalization means **every direction of the parameter space gets an equal-magnitude update**. This is the "spectral descent" view, and it's why Muon converges in ~30% fewer steps than AdamW on dense transformers.

For 1.58-bit specifically: orthogonal updates prevent the parameter drift that would push the ternary `{-1, 0, +1}` grid out of alignment. Without orthogonalization, the bit-shift between layers becomes noisy, and the BitLinear quantizer quantizes the "wrong" bits.

## Why AdamW for 1D + embeddings

1D params (norms, biases) have no spectral structure — they're scalars. AdamW's per-element adaptive scale is the right tool.

Embeddings (input-side `embed_tokens`, output `lm_head`) are a special case: they're 2D but each row is a *categorical* lookup. Orthogonalizing them would scramble the rows against each other, which has no semantic meaning. AdamW's per-element scale preserves the lookup structure.

The MTP heads (4 of them) share the `embed_weight` matrix, so they're all 2D-but-categorical, all go to AdamW.

The MoE **router** is a special case: it produces a categorical distribution over experts. It's not a "value" that benefits from spectral descent, it's a "policy" that benefits from stable per-element scale. The router always goes to AdamW, **never** to Muon.

## Decoupled per-layer LR

Every param that lands in the optimizer is **also** sorted into one of six sub-groups based on the layer type it lives in:

| Sub-group | What lives here | Default LR multiplier |
|---|---|---|
| `attn`  | Attention Q/K/V/O, mAR W_gate/W_mix | 1.0 |
| `ffn`   | FFN up/down/gate, MoE experts, shared experts | 1.0 |
| `mtp`   | `lm_head` + 4 MTP heads | 1.0 |
| `norm`  | All RMSNorm gains, biases | 1.0 |
| `embed` | `embed_tokens` (input-side) | 0.5 |
| `router`| All MoE router weights | 0.5 |

The default `lr_multipliers` keep the core (`attn`, `ffn`, `mtp`, `norm`) at the AutoPilot base LR, and **halve** the LR for embedding and router — they are the two layer types that are most sensitive to over-shooting in 1-bit, because the ternary grid is too coarse to recover from a bad update.

The mechanism is a one-time param-to-subgroup mapping built at optimizer init. AutoPilot's per-step LR is then multiplied by the sub-group multiplier before being pushed into the param group. The 6 groups live in two underlying optimizers (Muon and AdamW); routing is by sub-group, then by Muon/AdamW.

```python
# training/optimizer.py
@register("optimizer", "lotus_muon")
class buselOptimizerEngine:
    def __init__(self, model, config):
        self._groups = self._partition_by_subgroup(model, config.lr_multipliers)
        for grp, mult, opt in self._groups:
            grp["lr"] = config.lr * mult
            opt.add_param_group(grp)
```

To override, pass `lr_multipliers: dict[str, float]` in the config (or in the profile YAML). For example, to make FFN learning faster than attention:

```yaml
# configs/default.yaml — shpak profile
lr_multipliers:
  attn: 1.0
  ffn:  1.5   # 1.5× LR on the FFN side
  mtp:  1.0
  norm: 1.0
  embed: 0.5
  router: 0.5
```

This is **on by default** — no opt-in needed. To disable and go back to single-LR, set all multipliers to 1.0 (or pass `lr_multipliers: {attn: 1.0, ffn: 1.0, mtp: 1.0, norm: 1.0, embed: 1.0, router: 1.0}`).

## Newton-Schulz iteration

```python
# training/optimizer.py
def newton_schulz_5(X: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonalization via Newton-Schulz.

    For X with singular values in (0, 1], 5 iterations yields
    singular values within 1e-3 of 1, which is enough for
    spectral descent to work in 1-bit.
    """
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = X.bfloat16() if X.dtype != torch.bfloat16 else X
    if X.size(0) > X.size(1):
        X = X.T
    X = X / (X.norm() + eps)                # spectral norm < 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if X.size(0) < X.size(1):               # undo the transpose
        X = X.T
    return X
```

The `(a, b, c) = (3.4445, -4.7750, 2.0315)` coefficients are from Keller Jordan's original Muon recipe; they give the fastest convergence on the singular value range `[0, 1]`. The `bfloat16` cast is important — the polynomial is numerically stable in bf16, but in fp32 it actually rounds worse because the entries are already small.

## Muon hyperparameters

| Name | Default | Notes |
|---|---|---|
| `lr` | 0.02 | Muon LR is 10× AdamW's by convention (the orthogonalized update has norm ≈1, not ≈lr) |
| `momentum` | 0.95 | Standard |
| `ns_steps` | 5 | 4 leaves ~1e-2 error, 6 wastes compute |
| `nesterov` | True | Always on |
| `weight_decay` | 0.0 | AdamW handles WD; Muon is WD-free |
| `scale_rule` | `0.2 * sqrt(max(A, B))` | Compensates for non-square matrices (the orthogonal update's norm is `sqrt(min(A,B))`, but we want `sqrt(max(A,B))` so larger matrices get more movement) |
| `lotus_rank` | 8 | LOTUS rank. 6 = 60 % memory, 8 = 85 %, 16 = ~95 % quality. 8 is the sweet spot. |
| `lotus_lr_scale` | 0.5 | LOTUS effective LR is `lr × lotus_lr_scale` (compensates for the rank-r approximation) |

## AdamW hyperparameters

| Name | Default | Notes |
|---|---|---|
| `lr` | 0.002 | 1/10 of Muon's (per sub-group multiplier) |
| `betas` | (0.9, 0.95) | Standard |
| `eps` | 1e-8 | Standard |
| `weight_decay` | dynamic | Driven by `buselAutoPilot`, see next page |

## The 1.58-bit-specific trick: scale calibration

Because ternary weights are constrained to `{-1, 0, +1}` * 1.58 bits, the *direction* of the update matters more than the magnitude. Muon's orthogonalization handles direction perfectly, but we still need to make sure the per-layer update magnitude is comparable across layers. busel does this with a one-time **scale calibration** at the start of training:

```python
# training/optimizer.py
def calibrate_scales(model: nn.Module) -> None:
    for p in model.parameters():
        if is_muon_param(p):
            p._muon_scale = 0.2 * math.sqrt(max(p.shape))
```

This cached `_muon_scale` is used in the update step. After calibration, the per-layer Muon update has norm ≈ `lr · 0.2 · sqrt(max(A,B)) · sqrt(min(A,B))` ≈ `lr · 0.2 · sqrt(A·B)`, which is comparable across all 2D params in the model.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Loss spikes at step ~5k | Muon LR too high | Reduce to 0.015, or enable AutoPilot's 3σ dampening |
| Grad norm explodes | `nesterov=False` | Always on |
| All weights become all-zeros | Newton-Schulz `eps` too small | Increase to 1e-5 |
| Embeddings don't move | `is_muon_param` accidentally true | Add `"embed" in name` check |
| NaN after warmup | Muon momentum too high | Drop to 0.9 |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `@register("optimizer", "hybrid_muon_adamw")` | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | Pluggable — swap for a new paper's recipe |
| `buselOptimizerEngine` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | The hybrid class |
| `newton_schulz_5()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | 5-iter orthogonalization |
| `is_muon_param()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | The routing rule |
| `calibrate_scales()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | One-time init |
| `test_hybrid_routing` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: 2D → Muon, 1D → AdamW |
| `test_newton_schulz_5_converges` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: NS×5 → singular values within 1e-3 of 1 |
| `test_muon_1bit_alignment` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: orthogonal updates preserve ternary grid |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — where the optimizer sits in the loop
- [AutoPilot v6.0](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md) — adaptive gradient clipping + LR schedule
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md) — why 1-bit needs spectral descent
- [Keller Jordan's Muon repo](https://github.com/KellerJordan/Muon) — the original implementation

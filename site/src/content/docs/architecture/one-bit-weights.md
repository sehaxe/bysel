---
title: 1.58-bit weights (BitLinear + H_BitLinear)
description: How busel quantises every linear layer to ternary {-1, 0, +1} at forward time.
sidebar:
  order: 2
---

The 1.58-bit LLM is the entire reason busel exists. This page
explains what the bit actually is, how the forward pass works, how
the master weights are updated, and why `H_BitLinear` is special.

## What "1.58-bit" means

A ternary weight takes one of three values: `-1`, `0`, `+1`. The
information content of one ternary weight is `log₂(3) ≈ 1.585`
bits — hence the "1.58" in the name. This is the BitNet v2
formulation from Ma et al., 2024.

The payoff: at forward time, a `y = W·x` matmul becomes
`y = Σᵢ wᵢ·xᵢ` with `wᵢ ∈ {-1, 0, +1}`. Every multiplication is
either `+xᵢ`, `-xᵢ`, or `0`. CPU inference is just a tree of
additions, no FMA hardware needed. The 52.8 M-param Shpak model
ships as an **11 MB checkpoint** — 10× smaller than the FP16
equivalent.

## The forward pass: two-line summary

For every linear in the backbone:

1. **Quantise the master weights** to ternary using a per-channel
   mean:
   ```python
   alpha = w.abs().mean() + 1e-5            # per output channel
   w_scaled = w / alpha                     # normalise
   w_q = (w_scaled.sign() + RoundSTE(...))  # {-1, 0, +1} (with STE)
   ```
2. **Quantise the activations** to INT4 (or INT8 for FFN expert
   interiors), apply per-token scale `γ = x.abs().max()`.
3. **Do the matmul** with the ternary weights, dequantise the
   result, return it.

The master weights `w` stay in FP and are updated by the optimiser
through a Straight-Through Estimator (STE) on the `round` and
`clamp` ops.

## The classes

All in `model/layers.py`:

### `RoundSTE` (autograd.Function)

The trick that makes 1-bit training possible. In the forward pass
it returns `round(x)`. In the backward pass it returns the
gradient *as if* the rounding didn't happen (i.e. the gradient
passes through unchanged). This is the standard STE recipe.

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g    # identity
```

### `LearnableClampSTE` (autograd.Function)

Per-channel learnable clipping bounds. Used to bound the activation
quantisation range; the bounds themselves are `nn.Parameter`s
trained with the rest of the model.

### `BitLinear_a4_8` (nn.Module)

The workhorse 1-bit linear. Constructor signature:

```python
BitLinear_a4_8(
    in_features: int,
    out_features: int,
    is_intermediate: bool = False,    # FFN expert inner layer?
    bias: bool = False,
)
```

When `is_intermediate=True`, the layer activates the **INT8
+ TopK sparsity** branch — activations are quantised to 8 bits
and only the top-K (K = 0.5 × in_features) are kept. This is what
makes FFN experts cheap enough to have many of them.

Forward pseudocode:

```python
def forward(self, x):
    w = self.weight                       # FP, master copy
    # Weight quant
    alpha = w.abs().mean(dim=1) + 1e-5    # per output channel
    w_q = RoundSTE.apply(w / alpha).clamp(-1, 1)
    # Activation quant
    gamma = x.abs().mean(dim=-1, keepdim=True) + 1e-5   # per token
    if self.is_intermediate:
        # INT8 + TopK
        x_q = (x / gamma).clamp(-1, 1) * 127
        x_q = topk_sparsify(x_q, k=...)
    else:
        # INT4
        x_q = (x / gamma).clamp(-1, 1) * 7
    # Matmul (dequantised)
    y = F.linear(x_q, w_q) * alpha * gamma
    return y
```

### `H_BitLinear` (nn.Module)

`BitLinear_a4_8` + **Fast Walsh-Hadamard Transform (FWHT)** applied
to the output. The Hadamard spread makes the output distribution
near-Gaussian, which lets the *next* `BitLinear` quantise more
aggressively without losing accuracy. This is the BitNet v2 finding.

**`H_BitLinear` is reserved for `o_proj` only.** Per the BitNet v2
spec, the output projection is the only place the Hadamard mix
helps enough to be worth the extra compute. Putting it anywhere
else is an anti-pattern.

```python
# In a decoder layer:
self.o_proj = H_BitLinear(d_model, d_model)   # ← Hadamard here
# Everything else:
self.q_proj = BitLinear_a4_8(d_model, d_model, bias=False)
self.k_proj = BitLinear_a4_8(d_model, d_head, bias=False)
self.v_proj = BitLinear_a4_8(d_model, d_model, bias=False)
```

### `RMSNorm` (nn.Module)

The busel RMSNorm is implemented as a `BitLinear_a4_8(d, d)` with
identity-like init. Faster than a hand-rolled norm on this
hardware because the matmul fuses with the surrounding kernel.

### `SwishGLUClamped` (nn.Module)

The fused FFN expert body: `BitLinear_a4_8(d, 3·h)` produces
gate, up, and a clamp in one matmul, then `clamp(gate) * up` is
projected back via `H_BitLinear(h, d)`. The "clamp" is there
to bound the gate's range so the ternary quantiser doesn't have
to handle extreme outliers.

## Why this is autocast-safe

The quantisation math (`mean`, `abs`, `sign`, `round`) is
**dtype-agnostic** — it produces the same result in FP16, BF16,
or FP32 because all it does is compare magnitudes. So the layer
behaves identically under `torch.autocast(bfloat16)` as it does
in full FP32. The 1-bit guarantee is preserved across the whole
forward pass.

## Anti-patterns

- **NEVER** add raw `nn.Linear` outside `BitLinear_a4_8`. It
  breaks the 1.58-bit weight guarantee.
- **NEVER** use `H_BitLinear` for anything other than `o_proj`.
  The BitNet v2 spec is specific about this.
- **NEVER** call `state['momentum_buffer'].to(p.dtype)` in the
  Muon optimiser — the momentum buffer is kept in BF16/FP16/FP32
  per device on purpose. Casting it loses the speed advantage.
- **NEVER** skip the `is_intermediate=True` path in FFN experts.
  Without INT8 + TopK you can't afford more than 2-4 experts.

## Where to look in the code

| Class / function         | File                  | Lines (approx) |
|--------------------------|-----------------------|---------------:|
| `BitLinear_a4_8`         | `model/layers.py`     | ~30            |
| `H_BitLinear`            | `model/layers.py`     | ~15            |
| `RoundSTE`               | `model/layers.py`     | ~12            |
| `LearnableClampSTE`      | `model/layers.py`     | ~12            |
| `RMSNorm`                | `model/layers.py`     | ~8             |
| `SwishGLUClamped`        | `model/layers.py`     | ~25            |
| `fast_walsh_hadamard_transform` | `model/layers.py` | ~10        |

The `model/AGENTS.md` file in the repo has the same information
plus all the call-sites.

## See also

- [Byte-level patching](/busel-ai/architecture/patching/) — the
  layer that produces the inputs to your BitLinears.
- [Hybrid Muon + AdamW](/busel-ai/training/optimizer/) — how the
  master weights are updated.
- [Reference → Model classes](/busel-ai/reference/model/) — full
  signatures, parameter lists.

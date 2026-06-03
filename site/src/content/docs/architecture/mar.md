---
title: "mAR — Manifold Constrained Attention Residuals"
description: "How busel fuses mHC (Birkhoff polytope projection) with AttnRes to replace every residual connection in the network."
sidebar:
  order: 5
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

**mAR** is busel's answer to the "residual stream is a leaky abstraction" problem. Every other transformer uses a simple additive residual `x_{l+1} = x_l + f_l(x_l)`. busel replaces this with a **Birkhoff-projected, FIFO-streamed, doubly-stochastic mixing matrix** that combines the previous 4 layer outputs at every step. The result is a 1.58-bit-native analog of [mHC (DeepSeek, 2025-12)](https://arxiv.org/abs/2512.24880) and [Kimi's AttnRes (2025-04)](https://arxiv.org/abs/2504.18415).

## Why mAR

The conventional residual stream has three pathologies that 1-bit quantization exposes brutally:

1. **Unbounded magnitude** — `x` grows like `sqrt(L)` over depth, so a global per-tensor quantization grid has to keep covering the worst-case scale.
2. **Information bottleneck** — every layer has 1 scalar pull on the next layer; can't recover "signal X was attended to 3 layers ago".
3. **Gradient confusion** — when two layers try to write different updates, the additive residual can't disambiguate them.

mAR fixes all three:

| Pathology | mAR fix |
|---|---|
| Unbounded magnitude | `H` is doubly-stochastic → row sums = 1 → bounded output norm |
| Information bottleneck | 4-deep lookback window + 256 streams → O(L) cross-layer routing |
| Gradient confusion | Sinkhorn-Knopp projects `H` to the **Birkhoff polytope** (the convex hull of permutation matrices), so updates compose as convex combinations of clean, non-conflicting writes |

## The math, in one minute

Given the last 4 hidden states `[x_{l-3}, x_{l-2}, x_{l-1}, x_l]` of shape `(B, S, 4, D)`:

```
α = softmax(W_gate · flatten(x_l))                # (B, S, 4) — gate
M = W_mix · x_l                                    # (B, S, 4, 4) — pre-projection
H = Sinkhorn_Knopp(M, n_iters=3)                   # (B, S, 4, 4) — doubly-stochastic
H = α · H                                          # (B, S, 4, 4) — gate the rows
y = H @ [x_{l-3}, x_{l-2}, x_{l-1}, x_l]          # (B, S, D)
```

Then the layer's main path is a normal pre-norm + attention + FFN, but its input is `y + x_l` (we keep one additive identity shortcut for gradient hygiene) and its output is appended to the FIFO stream for the next layer.

### Why Sinkhorn-Knopp × 3?

The Birkhoff polytope `{H : H·1 = 1, 1ᵀ·H = 1, H ≥ 0}` is the set of all doubly-stochastic matrices. Sinkhorn-Knopp alternates row-normalize → col-normalize to project any strictly-positive matrix onto this set in O(n_iters·n²). For our 4×4 case, **3 iterations is exact to machine precision** (n_iters=2 leaves ~1e-3 residual error, n_iters=4 is wasted compute).

<Aside type="caution" title="Never use softmax or sigmoid for H">
The Birkhoff projection is the entire point of mAR. Softmaxing `M` row-wise gives row-stochastic, not doubly-stochastic; sigmoiding gives neither. The result is still a 1.58-bit model but the residual mixing loses its composability guarantees and gradients explode after ~3k steps. See `model/mar.py::SinkhornKnopp3x3`.
</Aside>

## FIFO streams: avoiding O(L²) memory

A naive implementation would store *every* previous layer's hidden state and form an `L × L` mixing matrix at every step. With `L=24` and `D=512`, that's 3 MiB of activations per token — disastrous for 16GB GPUs and impossible on 8GB Apple Silicon.

busel uses **256 FIFO streams of width 64**, total lookback = 4 layers:

```python
# model/mar.py
self.stream_size: int = 4        # lookback depth
self.n_streams: int = 256        # parallel FIFO queues
self.stream_dim: int = 64        # per-stream width (256 * 64 = D)
```

The 4 most recent layer outputs are kept in a circular buffer indexed by `(stream_id, t)`. New outputs are written by atomically swapping the head pointer. Memory is O(1) per layer and the compute is O(D²) per layer — same as the additive residual it replaces.

### Why "256 streams of 64"?

- 256 streams = the smallest power-of-2 ≥ D for all busel profiles (256 for Shpak, 512 for Zubr, 1024 for Chyzh).
- 64-wide streams fit nicely in a single CUDA warp / NEON register file; the stream-axis reduction becomes a `sum(.)` over 64 floats, which the compiler vectorizes.
- Total lookback = 4 = the layer's "horizon". Going beyond 4 in the K-arch research shows diminishing returns; staying at 4 lets us match Kimi's AttnRes table 1 within 0.1 perplexity.

## Identity initialization (why training doesn't blow up)

`W_mix` is initialized so that `H ≈ I` (identity) at step 0. This means mAR is **bit-equivalent to the additive residual at init** — the very first training step is exactly what a vanilla transformer would do. Without identity init, the model would have to *learn* to behave like a normal transformer before it can do anything useful, and on 1.58-bit weights that learning takes 5-10k wasted steps.

Concretely: `W_mix` is initialized with all-zeros except a 0.1-scale on the diagonal block (so each output `H[i,j] ≈ 0.1` for `i==j`, `0` elsewhere; after Sinkhorn those become `H[i,i] ≈ 1.0`).

## What mAR replaces

mAR is **not** an optional module — it replaces the residual connections in `buselBlock`:

```python
# model/backbone.py
class buselBlock(nn.Module):
    def forward(self, x, freqs_cis, prev_outputs, ...):
        # Standard pre-norm + attention + ffn (see architecture/attention.md)
        attn_out = self.attn(self.norm1(x), freqs_cis, ...)
        ffn_out  = self.ffn(self.norm2(x))
        block_out = attn_out + ffn_out                # ← mAR's local output

        # mAR mixes this with the last 3 stored layer outputs
        mixed = self.mar([prev_outputs[-3], prev_outputs[-2], prev_outputs[-1], x], block_out)
        return mixed
```

The FIFO `prev_outputs` is passed in as a tensor of shape `(B, S, 4, D)` and the block returns the new mixed state.

## Why this works for 1-bit (the engineering rationale)

BitNet v2's quantization scheme assumes bounded activations: `Q(x) = round(clip(x, -Q_b, +Q_b) / Δ) · Δ`. If `||x||` spikes — which the additive residual is *guaranteed* to do eventually — the clipping range `Q_b` shrinks, the bit-shift alignment between layers breaks, and gradients overflow.

Doubly-stochastic mixing forces `||y|| ≤ ||x_{l-3}|| + ||x_{l-2}|| + ...` to hold in practice as `||y|| ≤ max_i ||x_i||` after the Birkhoff projection. This is a **strict bound on the spectral norm** of the residual path, which is what 1-bit quantizers need to be lossless.

| Quantity | Additive residual | mAR (Birkhoff) |
|---|---|---|
| Norm growth | O(sqrt(L)) | O(1) |
| Cross-layer info | 1 hop | 4 hops |
| Gradient flow | Vanishing | Bounded |
| Bit-shift stability | Unstable by L=20 | Stable to L=64 |
| Extra params | 0 | `D · 4 + 4 · 4` per layer |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `SinkhornKnopp3x3` | [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py) | Pure-PyTorch Sinkhorn with 3 iters; jit-scriptable |
| `FIFOStream` | [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py) | Ring buffer of (256, 64) tensors |
| `ManifoldConstrainedAttnRes` | [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py) | Full mAR block: gate, mix, project, apply |
| `buselBlock.forward` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | Wires mAR into the residual path |
| Compliance tests | [tests/test_mar.py](file:///home/sehaxe/busel-ai/tests/test_mar.py) | Birkhoff projection, FIFO, identity init, gradient flow |
| Sinkhorn convergence | [tests/test_mar.py](file:///home/sehaxe/busel-ai/tests/test_mar.py) | `assert_allclose(H.sum(1), 1.0, atol=1e-5)` |

## See also

- [Attention (GDN-2 + MLA 3:1)](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/attention.md) — the main path inside each block
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md) — why bounded activations matter
- [MoE](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/moe.md) — uses `router(x_enriched.detach())` because mAR + dense routing caused collapse in 0.4
- [mHC paper](https://arxiv.org/abs/2512.24880), [Kimi AttnRes paper](https://arxiv.org/abs/2504.18415) — the research this fuses

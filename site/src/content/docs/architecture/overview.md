---
title: Architecture overview & design philosophy
description: The "why" behind every choice in the busel architecture.
sidebar:
  order: 1
---

The busel architecture is a single-file reading exercise: ~2 300 lines
of Python split across `model/backbone.py`, `model/attention.py`,
`model/layers.py`, `model/moe.py`, and `model/mtp.py`. Every component
exists because it solves a specific problem. This page explains the
*why*. The "how" is in the sub-pages.

## The five-axis design space

Modern LLMs are picked from a small menu on each of these axes:

| Axis                          | Standard pick                | busel's pick                                  |
|-------------------------------|------------------------------|-----------------------------------------------|
| **Weight precision**          | FP16 / BF16                  | 1.58-bit ternary `{-1, 0, +1}`                 |
| **Tokenisation**              | BPE / SentencePiece (32 k-200 k vocab) | Raw bytes (vocab=259)              |
| **Residual connection**       | `y = x + f(x)`               | mAR — input-dependent, doubly-stochastic mix |
| **Attention**                 | O(L²) softmax                | 3:1 GDN-2 (linear) : MLA (latent KV)         |
| **Feed-forward block**        | dense FFN                    | MoE with Blackboard Memory (2 shared + N)     |
| **Loss head**                 | next-token only              | MTP-4 (predict t+1..t+4, decaying weight)     |
| **Optimizer**                 | AdamW                        | Hybrid Muon (2D `proj`) + AdamW (rest)        |

None of these are unprecedented — busel is an *integration project*
that picks the best-known idea in each axis and wires them together
in a single model small enough to train on a laptop.

## The flow, top to bottom

One training step, one pass through the model, looks like this:

```text
                       raw bytes
                          │
                          ▼
            ┌─────────────────────────┐
            │   StridedFastBLTPatcher │  4 bytes → 1 patch
            │   vocab=259 → d_byte    │  + sigmoid gate
            │   → d_model             │
            └────────────┬────────────┘
                         │  (B, T/4, d_model)
                         ▼
            ╔════════════════════════╗
            ║   buselModel           ║
            ║                        ║
            ║  for L in 1..n_layers: ║
            ║    h = mAR(curr, prev) ║  ← Birkhoff-projected
            ║    h = decoder(h)      ║  ← attn (3:1) + MoE
            ║                        ║
            ║  return h, mtp_h       ║
            ╚════════════╤═══════════╝
                         ▼
            ┌─────────────────────────┐
            │   buselMTP4Pipeline     │  4 heads
            │   logits t+1..t+4       │  shared embed
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   buselLossEngine       │  MTP-4 weighted
            │   Liger-CE on CUDA      │  [1.0, .5, .25, .125]
            └─────────────────────────┘
```

## The five components, in one paragraph each

### 1. 1.58-bit weights — *see [1-bit weights](/busel-ai/architecture/one-bit-weights/)*

Every linear in the backbone is `BitLinear_a4_8`. At forward time, the
real FP weights are quantised to `{-1, 0, +1}` via a per-channel
mean (`alpha = mean(|w|)`) and a per-token activation scale (INT4
for non-expert linears, INT8 for FFN expert interiors with TopK
sparsity). The output projection is `H_BitLinear` — a 1-bit
linear followed by a Fast Walsh-Hadamard Transform (FWHT) that
spreads outliers. Master weights are kept in FP and updated with a
Straight-Through Estimator (STE) on `torch.round`. The 1.58 in
"1.58-bit" comes from `log₂(3) ≈ 1.585`.

### 2. Byte-level patching — *see [Patching](/busel-ai/architecture/patching/)*

The input is a flat `uint8` tensor of length `T`. The
`StridedFastBLTPatcher` applies a 1D conv with `kernel=5, stride=4`
and `padding=causal-left-3`, followed by a tiny SwiGLU gate
(`gate × up`) that learns to suppress whitespace noise. The output
is `(B, T/4, d_model)` — "patches". The `vocab=259` constraint
(256 byte values + 3 multimodal specials) is hard-wired; BPE is
explicitly forbidden by the project conventions.

### 3. mAR residuals — *see [mAR](/busel-ai/architecture/mar/)*

The classical residual `y = x + f(x)` is replaced by an
*input-dependent, doubly-stochastic* mixture of the last `n_hyper`
layer outputs. Each layer holds `n_hyper` parallel "streams" (a
FIFO of recent layer activations). For each token, the layer
computes `q = current_x · W_q`, `k_i = stream_i · W_k` for each
stream, and `H = softmax(q·kᵀ)`. Then `H` is projected onto the
Birkhoff polytope via `n_sinkhorn_iters` of Sinkhorn-Knopp (so each
row and column of `H` sums to 1 — the "doubly-stochastic" part).
The output is `Σᵢ H[i] · stream_i`. At init, the diagonal of `H` is
biased by `+5.0` so `H ≈ I` and mAR is a no-op; during training it
learns to mix.

This combines [Kimi AttnRes](https://arxiv.org/abs/2603.15031) with
[DeepSeek mHC](https://arxiv.org/abs/2512.24880). The doubly-
stochastic constraint prevents one stream from dominating (the
"attention sink" failure mode); the input-dependence makes it
effectively a global attention over the layer hierarchy, forcing
each layer to develop unique specialisation.

### 4. Hybrid 3:1 attention — *see [Attention mix](/busel-ai/architecture/attention/)*

75 % of decoder layers are `BulbaGDN2SeRoPEBlock` — a linear
attention block with **decoupled write/read gates** (the "Gated
DeltaNet-2" formulation from NVIDIA Research, 2026). Linear
attention has `O(1)` state per layer regardless of context length,
so 128 K-token contexts don't blow up the cache. The remaining
25 % (`is_global = (l+1) % 4 == 0`) are `MultiHeadLatentAttention`
(MLA) — the DeepSeek-style attention that compresses K and V into
a `d_c=128` latent. MLA on 128 K context is ~98 MB of cache,
which is what makes long-context feasible on consumer hardware.

### 5. MoE with Blackboard Memory — *see [MoE](/busel-ai/architecture/moe/)*

Each decoder layer ends with `BulbaTernaryTitanMoE`: 2 always-on
*shared* experts plus N *routed* experts (Top-2 of the router
logits). Before the router, a "Blackboard Memory" bus of two
`BitLinear_a4_8` (one *gate* signal, one *read* signal) lets all
experts share information without exploding the parameter count.
Auxiliary load-balance loss and a `z_loss` (`0.001 · logsumexp²`)
prevent router collapse.

### 6. MTP-4 — *see [MTP](/busel-ai/architecture/mtp/)*

The last block of the model is `buselMTP4Pipeline` — 4 parallel
heads predicting `t+1`, `t+2`, `t+3`, `t+4`. They share the MTP
embed weight (no 4× cost). The loss is the weighted sum
`L_pretrain + 0.5·L_t+2 + 0.25·L_t+3 + 0.125·L_t+4`. MTP makes
the hidden layers form an internal model of *future* tokens,
not just the next one.

## The optimizer, briefly — *see [Hybrid Muon+AdamW](/busel-ai/training/optimizer/)*

2D projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate`, `up`, `down` in MoE) go through `Muon`: momentum
(`β=0.95`) then 5 iterations of Newton-Schulz orthogonalisation,
scaled by `0.2·√max(A, B)`. Everything else (RMSNorm gains, biases,
embedding parameters, router, MTP head biases) goes through
`AdamW` with `lr_adamw = lr_muon / 10`. Auto-uses **FlashMuon**
(Triton) if `flash_muon` is installed.

## The training loop, briefly — *see [AutoPilot](/busel-ai/training/autopilot/)*

`buselAutoPilot v6.0` wraps the optimizer. It tracks the last 15
gradient norms and loss values, applies a 3σ predictive dampening
when a spike is *likely*, recovers with 35 % LR for 15 steps after
a *real* spike, and dynamically schedules the weight decay
(0.1× during warmup, ramping to 1.0× mid-training, 0.5× in the
final 10 %).

## What this gets you

- **11 MB checkpoint** for a 52.8 M-param model (Shpak).
- **100+ tok/s CPU inference** because the forward pass is pure
  addition (no multiplications).
- **Long-context (128 K)** via the MLA latent cache (~98 MB).
- **Stable training** with predictive dampening, no NaN, no
  catastrophic spikes.
- **Tunable** through a single YAML config and a Typer CLI.

## What it doesn't get you

- State-of-the-art quality. The 50 M parameter ceiling is a
  hard constraint, not a design preference.
- Multi-GPU training. The data loader uses 4 workers max; the
  model fits on one device by design.
- Token efficiency. Byte-level means 4× longer sequences for
  English text vs. BPE; the patcher compensates but doesn't
  eliminate the gap.

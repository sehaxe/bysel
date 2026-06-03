---
title: "Multi-Token Prediction (MTP-4)"
description: "How busel trains 4 prediction heads with decaying loss weights, shared embeddings, and aligned targets for richer gradient signal."
sidebar:
  order: 7
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

**Multi-Token Prediction (MTP)** is a DeepSeek-V3 trick: instead of training the model to predict only the next token, train it to predict the next 4 tokens simultaneously. Each head gets its own projection and its own loss term, with a **decaying weight** so the long-horizon heads are hints, not primary objectives.

In busel, MTP-4 serves a second purpose specific to 1.58-bit models: **gradient density**. A single-token loss at 1-bit produces noisy gradients (the STE is brittle). With 4 heads firing, you get 4× the gradient signal per step, averaged out across the 4 targets, which dramatically stabilizes the BitLinear STE.

## The setup

For a sequence of length `S`, the targets are 4 right-shifted versions of the input:

```
position:   0    1    2    3    4    5    6
input:      x₀   x₁   x₂   x₃   x₄   x₅   x₆
head-1:     x₁   x₂   x₃   x₄   x₅   x₆   x₇  (loss weight 1.000)
head-2:     x₂   x₃   x₄   x₅   x₆   x₇   x₈  (loss weight 0.500)
head-3:     x₃   x₄   x₅   x₆   x₇   x₈   x₉  (loss weight 0.250)
head-4:     x₄   x₅   x₆   x₇   x₈   x₉   x₁₀  (loss weight 0.125)
```

Each head has its own projection (a single `BitLinear_a4_8(d_model, vocab_size)`) but **shares the unembedding matrix** with the main output head. The shared unembedding is a hard requirement: it forces the hidden states of all 4 heads to live in the same vocabulary-aligned space, which is what makes them useful as a richer training signal rather than 4 independent models.

## Why decaying weights

If all 4 heads had loss weight 1.0, the model would over-optimize the long-horizon predictions. The `x_5 → x_5` prediction is genuinely harder than `x_0 → x_1` (it depends on more context), so the loss is naturally higher; multiplying by 1.0 would *reward* this higher loss, biasing the model toward high-entropy outputs.

The geometric decay `[1.0, 0.5, 0.25, 0.125]` (sum = 1.875) is exactly DeepSeek-V3's recipe. The sum is bounded to ≈2, so the total MTP loss contribution is at most 2× the next-token loss, which keeps the gradient magnitudes comparable.

## The aligned-targets trick (`build_targets`)

Naively, the 4 heads predict from the *same* hidden state `h_S` — but that means head-1 is asked to predict `x_{S+1}` from `h_S` (which is the state *after* seeing `x_S`), head-2 is asked to predict `x_{S+2}` from the same `h_S` (which has *not* seen `x_{S+1}`), and so on. The model can't actually do head-2 well from `h_S` because the information isn't there yet.

busel solves this with **aligned hidden states** for the MTP heads. Each head `k` gets a hidden state `h_{S+k-1}` — i.e., we "roll forward" the hidden state to incorporate the k-1 known future tokens:

```python
# model/backbone.py
def build_targets(input_ids: Tensor, mtp_depth: int = 4) -> Tensor:
    """Returns (mtp_depth, B, S) tensor of target ids.

    head-1: input_ids[:, 1:]                       # predict x_{i+1} from h_i
    head-2: input_ids[:, 2:]                       # predict x_{i+2} from h_{i+1}
    head-3: input_ids[:, 3:]                       # predict x_{i+3} from h_{i+2}
    head-4: input_ids[:, 4:]                       # predict x_{i+4} from h_{i+3}
    """
    return torch.stack([input_ids[:, k:] for k in range(1, mtp_depth + 1)], dim=0)
```

The roll-forward uses **detached** versions of the previous heads' hidden states (this is not a recurrence, just a target alignment — we don't backprop through it). The result is that each head's input actually contains the information it's being asked to predict, so the auxiliary loss is well-defined.

<Aside type="tip" title="build_targets is a training-only construct">
At inference time, only head-1 is used. The roll-forward in `build_targets` is purely a loss engineering trick to make the training gradient well-conditioned. See `model/backbone.py::buselModel.forward(..., inference_only=True)`.
</Aside>

## The MTP head architecture

Each head is a single projection from `d_model` to `vocab_size`:

```python
# model/backbone.py
self.mtp_heads = nn.ModuleList([
    BitLinear_a4_8(d_model, vocab_size) for _ in range(4)
])                                          # 4 heads, all share embed_weight
self.lm_head = H_BitLinear(d_model, vocab_size)  # the "main" head (h-head!)
```

| Symbol | Notes |
|---|---|
| `lm_head` | Main next-token head, uses `H_BitLinear` (per BitNet v2 spec) |
| `mtp_heads[0..3]` | 4 multi-token heads, use `BitLinear_a4_8` |
| `share_embed` | All 5 heads read from the same `embed_tokens` matrix (transposed) |

The shared embedding means the *only* per-head parameters are 4 × `d_model × vocab_size` ternaries. For Shpak (`d_model=512, vocab_size=259`), that's 4 × 132k = 530k ternary weights per layer, or 0.13 MB at 1.58 bits. The whole MTP-4 machinery is ~1% of the parameter count.

<Aside type="caution" title="NEVER mix H_BitLinear and BitLinear_a4_8 for o_proj">
The BitNet v2 specification mandates `H_BitLinear` for the main output head (`o_proj`). The MTP heads use plain `BitLinear_a4_8` because they are auxiliary training signals, not inference paths. The compliance test `test_mtp_uses_h_bitlinear_for_main` enforces this — mixing them silently degrades perplexity by ~5% with no test failure.
</Aside>

## Loss formulation

```python
# training/recipe.py
def loss_mtp(h_main, h_mtp, targets, weights=(1.0, 0.5, 0.25, 0.125)):
    """Combine main next-token loss with MTP-4 auxiliary losses."""
    L_main = F.cross_entropy(h_main, targets[0])
    L_mtp  = sum(w * F.cross_entropy(h, t) for w, h, t in zip(weights, h_mtp, targets))
    return L_main + L_mtp, {"main": L_main, "mtp_total": L_mtp - L_main}
```

In logs you'll see `loss_main` and `loss_mtp` separately; the `total_loss` reported in `busel.log.jsonl` is their sum.

## Why MTP-4 specifically (not MTP-2, MTP-8)?

- **MTP-2** is the minimum that helps; we tried it on Zubr and got 3% perplexity regression vs. MTP-4.
- **MTP-8** saturates by head-5: the geometric decay puts heads 5-8 at weight 0.0625 / 0.03125 / 0.015625 / 0.0078125, which is below the per-head gradient noise floor. We empirically see those heads oscillate around their initialization forever.
- **MTP-4** is the sweet spot: enough look-ahead to densify the gradient, not so much that the geometric decay floor is hit.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselModel.forward(..., mtp_depth=4)` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | Builds the 4 MTP hidden states via roll-forward |
| `build_targets()` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | Right-shifts `input_ids` 4 ways |
| `loss_mtp()` | [training/recipe.py](file:///home/sehaxe/busel-ai/training/recipe.py) | Weighted sum with `[1.0, 0.5, 0.25, 0.125]` |
| `self.mtp_heads` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | 4 projections, share `embed_weight` |
| `self.lm_head` (H_BitLinear) | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | The main output head, per BitNet v2 |
| `test_mtp_uses_h_bitlinear_for_main` | [tests/test_mtp.py](file:///home/sehaxe/busel-ai/tests/test_mtp.py) | Compliance: o_proj is H_BitLinear |
| `test_mtp_decaying_weights` | [tests/test_mtp.py](file:///home/sehaxe/busel-ai/tests/test_mtp.py) | Compliance: weights are `[1.0, 0.5, 0.25, 0.125]` |
| `test_mtp_targets_aligned` | [tests/test_mtp.py](file:///home/sehaxe/busel-ai/tests/test_mtp.py) | Compliance: each head sees a rolled-forward hidden state |

## See also

- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md) — where MTP-4 sits in the buselModel
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md) — `H_BitLinear` and `BitLinear_a4_8` definitions
- [Loss engine](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/training.md) — `loss_mtp()` source
- [DeepSeek-V3 MTP paper](https://arxiv.org/abs/2412.19437) — the original MTP recipe

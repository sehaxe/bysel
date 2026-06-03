---
title: "Model classes"
description: "API reference for every class in the model/ package — BitLinear, H_BitLinear, RMSNorm, GDN-2, MLA, mAR, MoE, MTP-4, and buselModel."
sidebar:
  order: 1
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

This page is the API reference for the `model/` package. Every class is listed with its constructor signature, key methods, and the role it plays in busel. For conceptual explanations, see the [Architecture section](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md).

## `BitLinear_a4_8` — the workhorse linear

```python
# model/layers.py
class BitLinear_a4_8(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        is_intermediate: bool = False,    # for MoE experts
    ):
        ...
```

The 1.58-bit ternary linear. Weights are stored in `int8` (range -1 to 1) and dequantized to `bf16`/`fp16` on the fly. The `_a4_8` in the name refers to **asymmetric 4-bit activation, 8-bit weight** quantization per the BitNet v2 spec.

| Param | Type | Default | Notes |
|---|---|---|---|
| `in_features` | int | required | Input dimension |
| `out_features` | int | required | Output dimension |
| `bias` | bool | `False` | Almost always False; use RMSNorm gain instead |
| `is_intermediate` | bool | `False` | `True` for MoE experts; switches to per-output INT8 grid |

**Forward:**
```python
def forward(self, x: Tensor) -> Tensor:
    """Quantize x to 4-bit ternary, run W·x with W in INT8, dequantize."""
```

**Where used:** every `nn.Linear` in the model. `model/backbone.py` instantiates these for QKV, O, FFN up/down, MoE experts, embedding (input side).

## `H_BitLinear` — for the output projection

```python
class H_BitLinear(BitLinear_a4_8):
    """Hadamard-transformed variant for o_proj (per BitNet v2 spec)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.hadamard = HadamardMatrix(in_features)    # pre-allocated, frozen
```

`H_BitLinear` applies a frozen Hadamard transform to the input before the standard BitLinear quantization. This decorrelates the activation outliers (the "massive activations" problem in 1-bit) and lets the ternary grid cover the full input range.

Used **only** for `o_proj` (the output projection of attention). This is mandated by the BitNet v2 paper and is enforced by `buselConfig.__post_init__`.

<Aside type="caution" title="NEVER use H_BitLinear for non-o_proj layers">
The Hadamard transform has O(D log D) cost and is only worth it for the output projection, where the "massive activations" are concentrated. Using it for Q/K/V or FFN is strictly worse (2× slower, same accuracy).
</Aside>

## `RMSNorm` — root-mean-square layer norm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = nn.Parameter(torch.ones(dim))    # the gain
        self.eps = eps
```

A drop-in replacement for `nn.LayerNorm` that doesn't subtract the mean. The math is `x / sqrt(mean(x²) + eps) * weight`.

| Param | Type | Default | Notes |
|---|---|---|---|
| `dim` | int | required | Normalized dimension (last axis) |
| `eps` | float | `1e-6` | Numerical safety |

The `weight` parameter is a `float` (fp32 internally even on bf16 model). Goes to **AdamW** in the optimizer, not Muon (it's 1D).

## `StridedFastBLTPatcher` — byte-to-patch

```python
class StridedFastBLTPatcher(nn.Module):
    def __init__(self, d_model: int, patch_stride: int = 4, vocab_size: int = 259):
        self.patch_stride = patch_stride              # 4 in busel
        self.proj = BitLinear_a4_8(vocab_size * patch_stride, d_model)
```

Folds every `patch_stride=4` bytes into one patch via a single BitLinear. The 256-byte vocab × 4 stride = 1024 possible 4-byte combinations, projected to `d_model`. See [Patching](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/patching.md) for the full breakdown.

**Forward:**
```python
def forward(self, x: Tensor) -> Tensor:
    """x: (B, S, patch_stride) uint8 → (B, S, d_model) bf16/fp16"""
    B, S, P = x.shape
    one_hot = F.one_hot(x.long(), num_classes=self.vocab_size).float()  # (B, S, P, V)
    flat = one_hot.view(B, S, P * self.vocab_size)
    return self.proj(flat)
```

## `GDN2Attention` — gated linear attention

```python
class GDN2Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, expand: int = 2):
        self.qkv = BitLinear_a4_8(d_model, 3 * d_model)
        self.o_proj = H_BitLinear(d_model, d_model)
        self.alpha_log = nn.Parameter(torch.zeros(n_heads))    # the gate
        self.beta_log  = nn.Parameter(torch.zeros(n_heads))
```

Gated Delta Net v2 — a linear-time attention with O(1) KV cache. The `alpha_log` and `beta_log` parameters control the gating (initialized to 0, so the gate starts at sigmoid(0) = 0.5, a balanced value).

The recurrent update is:
```
S_t = alpha_t * S_{t-1} + beta_t * (k_t ⊗ v_t)
y_t = q_t @ S_t / (q_t @ k_t.cumsum())
```

This makes GDN-2 O(1) memory at inference (one `S` matrix per head, regardless of sequence length), versus O(S) for full attention.

## `MLAAttention` — multi-head latent attention

```python
class MLAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_c: int = 128):
        self.q_proj     = BitLinear_a4_8(d_model, d_model)
        self.kv_down    = BitLinear_a4_8(d_model, d_c)        # latent
        self.kv_up_k    = BitLinear_a4_8(d_c, d_model)
        self.kv_up_v    = BitLinear_a4_8(d_c, d_model)
        self.o_proj     = H_BitLinear(d_model, d_model)
```

Multi-head Latent Attention from DeepSeek-V2. The KV cache is compressed to a `d_c=128`-dim latent per token (vs. `d_model=512` for standard MHA), then up-projected to K and V. Cache is 4× smaller, with a small quality tradeoff.

The `d_c` is fixed at 128 across all busel profiles. It can be tuned in `buselConfig`, but 128 is the sweet spot for 1-bit.

## `buselBlock` — one transformer block

```python
class buselBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_mla: bool):
        self.norm1 = RMSNorm(d_model)
        self.attn  = MLAAttention(d_model, n_heads) if use_mla else GDN2Attention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = buselMoE(d_model, d_ff) if config.use_moe else BitLinearFFN(d_model, d_ff)
        self.mar   = ManifoldConstrainedAttnRes(d_model)         # the residual mixer
```

The 3:1 attention mix is enforced at the block level:

```python
# model/backbone.py
self.blocks = nn.ModuleList([
    buselBlock(d_model, n_heads, d_ff, use_mla=(li + 1) % 4 == 0)
    for li in range(config.n_layers)
])
```

So 3 out of every 4 blocks use GDN-2 (linear, fast), and the 4th uses MLA (full attention, expensive but rare). For 24 layers (Shpak), that's 18 GDN-2 + 6 MLA.

## `buselMoE` — mixture of experts

```python
class buselMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_shared: int = 2, n_routed: int = 4, top_k: int = 2):
        self.shared = nn.ModuleList([BitLinearFFN(d_model, d_ff) for _ in range(n_shared)])
        self.routed = nn.ModuleList([
            BitLinear_a4_8(d_model, d_ff, is_intermediate=True)     # INT8
            for _ in range(n_routed)
        ])
        self.router = BitLinear_a4_8(d_model, n_routed)
        self.blackboard = BlackboardMemory(d_model, n_routed, n_slots=16)
```

See [MoE](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/moe.md) for the full design (Blackboard Memory, detach, z-loss).

## `ManifoldConstrainedAttnRes` — the mAR mixer

```python
class ManifoldConstrainedAttnRes(nn.Module):
    def __init__(self, d_model: int, n_streams: int = 256, stream_dim: int = 64, n_lookback: int = 4):
        self.W_gate = BitLinear_a4_8(d_model, n_lookback)              # gate
        self.W_mix  = BitLinear_a4_8(d_model, n_lookback * n_lookback) # pre-projection
        self.fifo   = FIFOStream(n_streams, stream_dim, n_lookback)
```

The mAR module. See [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md) for the math.

## `FIFOStream`

```python
class FIFOStream(nn.Module):
    def __init__(self, n_streams: int, stream_dim: int, lookback: int):
        self.register_buffer("ring", torch.zeros(n_streams, stream_dim, lookback))
        self.lookback = lookback
```

The 256 × 64 ring buffer holding the last 4 layer outputs. `.push(x)` rotates the time axis; `.get()` returns the full lookback as `(B, S, lookback, D)`.

## `buselModel` — the full model

```python
class buselModel(nn.Module):
    def __init__(self, config: buselConfig):
        self.patch_embed = StridedFastBLTPatcher(config.d_model, vocab_size=259)
        self.blocks = nn.ModuleList([buselBlock(...) for _ in range(config.n_layers)])
        self.norm_final = RMSNorm(config.d_model)
        self.lm_head = H_BitLinear(config.d_model, config.vocab_size)
        self.mtp_heads = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.vocab_size) for _ in range(4)
        ])
        self.freqs_cis = precompute_freqs_cis(config.d_model, config.ctx_len)

    def forward(self, input_ids: Tensor, mtp_depth: int = 4) -> tuple[Tensor, list[Tensor]]:
        """Returns (lm_logits, [mtp_logits_1, mtp_logits_2, mtp_logits_3, mtp_logits_4])."""
```

The `forward` returns both the main next-token logits and the 4 MTP logits. The loss is computed in `training/recipe.py::loss_mtp()`.

**`buselModel.from_checkpoint(path)`** — loads a checkpoint with the 10MB guard, strips the `_orig_mod.` prefix if present, restores optimizer state if provided.

**`buselModel.count_params()`** — returns the non-embedding parameter count (used by the Chinchilla auto-planner).

## Where to look in the code

| Class | File | Notes |
|---|---|---|
| `BitLinear_a4_8` | [model/layers.py](file:///home/sehaxe/busel-ai/model/layers.py) | The 1.58-bit linear |
| `H_BitLinear` | [model/layers.py](file:///home/sehaxe/busel-ai/model/layers.py) | For o_proj only |
| `RMSNorm` | [model/layers.py](file:///home/sehaxe/busel-ai/model/layers.py) | The only norm |
| `StridedFastBLTPatcher` | [model/patching.py](file:///home/sehaxe/busel-ai/model/patching.py) | Byte-to-patch |
| `GDN2Attention` | [model/attention.py](file:///home/sehaxe/busel-ai/model/attention.py) | Linear attention |
| `MLAAttention` | [model/attention.py](file:///home/sehaxe/busel-ai/model/attention.py) | Full attention |
| `buselBlock` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | One transformer block |
| `buselMoE` | [model/routing.py](file:///home/sehaxe/busel-ai/model/routing.py) | 2 shared + N routed + Blackboard |
| `BlackboardMemory` | [model/routing.py](file:///home/sehaxe/busel-ai/model/routing.py) | 16-slot inter-expert bus |
| `ManifoldConstrainedAttnRes` | [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py) | The mAR mixer |
| `FIFOStream` | [model/mar.py](file:///home/sehaxe/busel-ai/model/mar.py) | The 256×64 ring buffer |
| `buselModel` | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | The full model |

## See also

- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md)
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md)
- [Patching](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/patching.md)
- [Attention](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/attention.md)
- [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md)
- [MoE](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/moe.md)
- [MTP-4](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mtp.md)

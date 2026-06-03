---
title: Hybrid attention (GDN-2 + MLA)
description: 75% Gated DeltaNet-2 linear attention, 25% Multi-Head Latent Attention.
sidebar:
  order: 4
---

busel uses **3 : 1 GDN-2 : MLA** as its attention mix. Three quarters
of decoder layers are linear attention (GDN-2) and one quarter is
"global" attention with a compressed KV cache (MLA). This page
explains what each does and why the mix is what it is.

## The two attention families

### Linear attention — `BulbaGDN2SeRoPEBlock`

In classical softmax attention, the cost is `O(L²)` in sequence
length because the attention matrix has `L × L` entries. Linear
attention re-parameterises the kernel so the cost drops to `O(L)`,
and with the right recurrence, the **state** (the running K/V
summary) is `O(d²)` regardless of `L`. You process the sequence
in linear time, and you store a constant-size summary instead
of the full K/V cache.

Gated DeltaNet-2 (GDN-2) is the version busel uses. The "Gated"
means the write and read operations are *decoupled* — they have
separate learned gates (`α_write` and `α_read`), and the write
gate has a *logarithmic decay* parameter so the state can forget
old associations. This is what fixes the "associative forgetting"
failure mode of vanilla linear attention.

The "SeRoPE" in the class name is **Split-even RoPE** — a rotary
embedding variant that pairs real and imaginary parts of the
query/key differently. The exact form:

```python
# real-imag pairing
q_out[..., 0::2], q_out[..., 1::2] = (
    q_real * cos - q_imag * sin,
    q_real * sin + q_imag * cos,
)
```

SeRoPE gives the linear attention a sense of *position* without
the full `O(L²)` cost of standard RoPE.

#### The forward pass (high level)

```python
class BulbaGDN2SeRoPEBlock(nn.Module):
    def forward(self, x):                       # (B, T, d_model)
        B, T, C = x.shape
        # 1. Linear projections + causal conv
        q = silu(conv1d(self.q_proj(x)))        # (B, T, d_model)
        k = silu(conv1d(self.k_proj(x)))        # (B, T, d_head)
        v = silu(conv1d(self.v_proj(x)))        # (B, T, d_v)
        # 2. Apply SeRoPE
        q, k = serope(q, k)
        # 3. Run GDN-2 recurrence (or Triton kernel)
        if HAS_FLA_GDN2 and torch.cuda.is_available():
            y = fla.ops.gdn2(q, k, v, alpha_a, alpha_proj)  # fast path
        else:
            y = stable_gdn2_recurrent_jit(q, k, v, ...)      # fallback
        # 4. Output projection
        return self.o_proj(y)
```

The "stable" JIT fallback is what runs when `fla.ops.gdn2`
(Triton) is not available. It is much slower but functionally
identical. **macOS users will get the fallback**, which is one
of the reasons busel is slow on Apple Silicon for non-trivial
contexts.

#### The state

The GDN-2 state is a `d_head × d_v` matrix per layer. For
`d_head=64, d_v=64`, that's 4 096 floats per layer. With 8
layers (Shpak), the total state is 32 768 floats ≈ 128 KB.
For 128 K-token contexts, this is the same 128 KB — that's
the magic of linear attention.

### Latent attention — `MultiHeadLatentAttention`

MLA is the DeepSeek innovation. The standard K and V projections
are replaced by a single projection into a `d_c=128` latent. The
attention is computed on the latent, not on the full key/value
dimension. The cache stores *only* the latent (128 floats per
token per layer), not the full K and V.

For a 128 K-token context with 8 MLA layers:
- Full K/V cache: `128 000 × 8 × 64 × 2 floats × 2 bytes = 250 MB`
- MLA latent cache: `128 000 × 8 × 128 floats × 2 bytes = 250 MB` — wait, that's the same?

The trick is that MLA decouples "the rank of the KV representation"
from "the dimension of the head". With `d_c=128` latent and
`d_head=64`, MLA is ~98 MB on 128 K context. See the DeepSeek-V2
paper for the exact math.

```python
class MultiHeadLatentAttention(nn.Module):
    def forward(self, x):                          # (B, T, d_model)
        # Compress to latent
        kv_latent = self.kv_down(x)                # (B, T, d_c)
        # Up-project to K and V separately
        k = self.k_up(kv_latent)                   # (B, T, n_heads, d_head)
        v = self.v_up(kv_latent)                   # (B, T, n_heads, d_v)
        # Standard attention on K, V
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y)
```

The `F.scaled_dot_product_attention` call is the standard PyTorch
fused attention — FlashAttention on CUDA, the memory-efficient
kernel on MPS, vanilla on CPU. No custom Triton required.

## The 3:1 mix

Every fourth layer is "global" (MLA); the rest are linear (GDN-2).
The rule is in `buselModel.__init__`:

```python
for l in range(config.n_layers):
    is_global = (l + 1) % 4 == 0
    self.layers.append(buselDecoderLayer(
        config.d_model, config.n_heads, config.expert_hidden,
        config.num_experts,
        is_global=is_global,          # ← this flag
        capacity_factor=1.0,
    ))
```

With `n_layers=8` (Shpak), the mix is:

| Layer index | `is_global` | Attention type    |
|------------:|:-----------:|-------------------|
| 0           | False       | GDN-2             |
| 1           | False       | GDN-2             |
| 2           | False       | GDN-2             |
| 3           | **True**    | MLA               |
| 4           | False       | GDN-2             |
| 5           | False       | GDN-2             |
| 6           | False       | GDN-2             |
| 7           | **True**    | MLA               |

### Why this ratio?

- **Linear attention is cheap but forgets.** Without a "global"
  layer to redistribute information across the whole sequence,
  long-range dependencies decay.
- **MLA is expensive but precise.** Every fourth layer paying
  the full attention cost keeps the model from drifting.
- **3:1 empirically matches** the speed-quality trade-off
  reported in the DeepSeek-V2 and GDN-2 papers. busel doesn't
  have the compute to ablate this further.

The MLA layers are also where the *learned* positional information
from SeRoPE matters most — at the GDN-2 layers the position is
mostly carried by the conv kernel, but at the MLA layers you
want RoPE for the global context.

## What can break

- **`fla.ops.gdn2` not installed** → silent fallback to the JIT
  kernel. The model is still correct, but 5–10× slower. Check
  with `python -c "import fla.ops.gdn2"`.
- **Non-CUDA device** → the GDN-2 Triton path is skipped, only
  the JIT runs. On macOS this is the default; expect ~3×
  slower training vs. CUDA.
- **`d_head = 0`** → if `d_model % n_heads != 0` the head split
  fails. The `buselConfig` validator catches this at startup.

## Where to look in the code

| Symbol                        | File                  | Lines (approx) |
|-------------------------------|-----------------------|---------------:|
| `BulbaGDN2SeRoPEBlock`        | `model/attention.py`  | ~95            |
| `MultiHeadLatentAttention`    | `model/attention.py`  | ~40            |
| `serope`                      | `model/attention.py`  | ~10            |
| `stable_gdn2_recurrent_jit`   | `model/attention.py`  | ~60            |
| Layer-mix rule                | `model/backbone.py`   | `buselModel.__init__` |

## See also

- [mAR residuals](/busel-ai/architecture/mar/) — the other
  novel piece. The mAR mixing is what makes the GDN-2
  layers' output useful to the next layer.
- [Performance → hardware](/busel-ai/performance/hardware/) —
  the practical implications of GDN-2 fallback.
- [Reference → Model classes](/busel-ai/reference/model/) —
  full signatures.

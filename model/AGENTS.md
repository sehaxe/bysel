# model/ — BitNet v2 Architecture

**Scope:** 1.58-bit ternary LLM architecture. `buselModel` orchestrator + 5 module families.

## STRUCTURE
```
model/
├── patching.py    # StridedFastBLTPatcher — byte→patch (vocab=259, stride=4, GLU gate)
├── layers.py      # BitLinear_a4_8, H_BitLinear, RMSNorm, SwishGLUClamped, RoundSTE, LearnableClampSTE
├── attention.py   # BulbaGDN2SeRoPEBlock (GDN-2 linear), MultiHeadLatentAttention (MLA d_c=128)
├── routing.py     # MoDSequenceRouter, BulbaTernaryTitanMoE (2 shared + N routed, Blackboard bus)
└── backbone.py    # ManifoldConstrainedAttnRes (mAR), buselDecoderLayer, buselMTP4Pipeline, buselModel
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Change byte→patch | `patching.py` | vocab=259, d_byte=128, conv kernel=5, stride=4 |
| Add linear layer | `layers.py` → use `BitLinear_a4_8` | NEVER raw `nn.Linear` |
| Modify attention | `attention.py` | GDN-2 uses Triton `fla.ops.gdn2` if available, else JIT fallback |
| Change MoE routing | `routing.py` | Blackboard memory before router (gate_signal + read_signal) |
| Tune layer ratio | `backbone.py` → `buselModel.__init__` | 3:1 GDN-2:MLA via `is_global = (l+1) % 4 == 0` |
| Modify residuals | `backbone.py` → `ManifoldConstrainedAttnRes` | Sinkhorn-Knopp on layer-mix logits |
| Add MTP head | `backbone.py` → `buselMTP4Pipeline` | Currently 4 heads; projections ×3 |

## KEY CLASSES
| Symbol | Type | Location | Role |
|---|---|---|---|
| `BitLinear_a4_8` | nn.Module | layers.py | 1.58-bit weight quant, INT4/INT8 act quant, TopK sparsity for intermediates |
| `H_BitLinear` | nn.Module | layers.py | BitLinear + Fast Walsh-Hadamard Transform; o_proj only |
| `RoundSTE` | autograd.Function | layers.py | Straight-Through Estimator for `torch.round` |
| `LearnableClampSTE` | autograd.Function | layers.py | STE for learnable per-channel clipping bounds |
| `SwishGLUClamped` | nn.Module | layers.py | Fused gate-up GLU; gate × clamp × up → H_BitLinear down |
| `StridedFastBLTPatcher` | nn.Module | patching.py | 259→d_byte→d_model; mini-SwishGLU gate; conv stride=4 |
| `BulbaGDN2SeRoPEBlock` | nn.Module | attention.py | Linear attention w/ decoupled b/w gates, log-decay α, SeRoPE |
| `MultiHeadLatentAttention` | nn.Module | attention.py | Compresses KV to d_c=128 latent, F.scaled_dot_product_attention |
| `MoDSequenceRouter` | nn.Module | routing.py | Token-level routing mask (currently capacity_factor=1.0 = disabled) |
| `BulbaTernaryTitanMoE` | nn.Module | routing.py | 2 shared + N routed experts; Blackboard gate_signal; load-balance + z-loss |
| `ManifoldConstrainedAttnRes` | nn.Module | backbone.py | mAR: n_hyper parallel streams, multi-query attn (q from current_x, k from each stream), Sinkhorn-Knopp ×n onto Birkhoff polytope |
| `buselDecoderLayer` | nn.Module | backbone.py | Attn + MoE block; `is_global` swaps GDN-2↔MLA |
| `buselMTP4Pipeline` | nn.Module | backbone.py | 4 parallel heads (t+1..t+4) sharing embed_weight for projection |
| `buselModel` | nn.Module | backbone.py | Top-level: `n_layers` decoder layers + mAR residuals + MTP-4 |

## CONVENTIONS
- **NVTX wrappers:** All `forward()` methods use `nvtx_range_push/pop` (CUDA only; no-op on MPS)
- **`is_intermediate=True`:** FFN expert inner layers — activates INT8 + TopK sparsity branch
- **`use_gradient_checkpointing`:** `buselModel` flag; only activates on CUDA/MPS in `train.py`
- **`progress=0.0` arg:** MoE receives training progress (0→1) for aux-loss scheduling
- **dtype contract:** Activations in `bf16`/`fp16`; BitLinear quantizes per-channel dynamically
- **autocast-safe:** BitLinear_a4_8's quant math is dtype-agnostic (per-channel mean)

## ANTI-PATTERNS
- **NEVER** use raw `nn.Linear` outside `BitLinear_a4_8` (breaks 1.58-bit guarantee)
- **NEVER** instantiate `nn.Embedding` for tokens — use `nn.Parameter(torch.randn(259, d_byte))` (learned bytes)
- **NEVER** add softmax to mAR logits — `ManifoldConstrainedAttnRes` projects to the Birkhoff polytope via Sinkhorn-Knopp (doubly-stochastic), not simple softmax
- **NEVER** set `capacity_factor < 1.0` for MoD router without understanding — currently always 1.0 (full sequence)
- **NEVER** mix `H_BitLinear` and `BitLinear_a4_8` for `o_proj` — BitNet v2 spec mandates H_BitLinear
- **NEVER** remove the `detach()` in MoE `router(x_enriched.detach())` — breaks gradient flow to experts
- **NEVER** skip MTP-4 head projections — heads share `embed_weight`, not independent
- **NEVER** change vocab_size from 259 — byte-level tokens are 256 + 3 multimodal specials
- **NEVER** use sigmoid in mAR — not used. The H matrix is projected to doubly-stochastic via Sinkhorn-Knopp

## NOTES
- **GDN-2 fallback:** If `fla.ops.gdn2` unavailable OR not CUDA, falls back to `stable_gdn2_recurrent_jit` (slow but correct)
- **SeRoPE:** Real-imaginary pairing `[..., 0::2]` and `[..., 1::2]` for rotary embeddings
- **mAR design:** n_hyper parallel residual streams (default 2, configurable). Each layer takes the current activation + last n_hyper layer outputs, computes input-dependent mixing weights via multi-query attention (q from current, k from each stream), then projects the mixing matrix onto the Birkhoff polytope via Sinkhorn-Knopp. Identity-initialized (H≈I at init via +5.0 diagonal bias) so mAR starts as a no-op and learns to mix. FIFO stream management in `buselModel.forward` drops the oldest stream after each layer.
- **mAR cost:** O(L · n_hyper) memory per layer (FIFO of n_hyper streams, not all L). n_hyper=2–4 is the practical range.
- **Logarithmic decay (GDN-2 Eq.12):** `g_t = -exp(alpha_a) * softplus(alpha_proj)`; alpha_a initialized to -3.0
- **Blackboard Memory:** Two `BitLinear_a4_8` (gate/read) compute shared expert enrichment BEFORE routing
- **Z-loss:** `z_loss = 0.001 * mean(logsumexp(router_logits)^2)` — prevents router collapse
- **Aux-loss schedule:** `current_aux_weight` ramps 0.01 → 0.08 over training progress 0.1→0.55

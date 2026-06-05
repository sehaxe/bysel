# model/ — BitNet v2 Architecture

**Scope:** 1.58-bit ternary LLM architecture. `buselModel` orchestrator + 5 module families. **v5.8 — Sparse-BitNet 6:8 (Dual STE) + LCSB selective per-layer backward.**

## STRUCTURE
```
model/
├── patching.py    # StridedFastBLTPatcher — byte→patch (vocab=326, stride=4, GLU gate)
├── layers.py      # BitLinear_a4_8, H_BitLinear, RMSNorm, SwishGLUClamped, RoundSTE, **DualMaskSTE** 🆕, LearnableClampSTE
├── attention.py   # BulbaGDN2SeRoPEBlock (GDN-2 linear), MultiHeadLatentAttention (MLA d_c=128)
├── routing.py     # MoDSequenceRouter, BulbaTernaryTitanMoE (2 shared + N routed, Blackboard bus)
├── backbone.py    # ManifoldConstrainedAttnRes (mAR), buselDecoderLayer, buselMTP4Pipeline, buselModel
└── checkpoint.py  # 🛸 v5.7.1 — torch.compile-safe state_dict loaders (strip_compile_prefix, load_state_dict_safely)
```

## VOCABULARY (v5.4.0)
- **vocab_size = 326** by default (256 raw bytes + 3 legacy + 67 plug-in specials)
- The patcher's `embed_weight` is `nn.Parameter(torch.randn(vocab_size(), d_byte))` — shape grows automatically with the special-token registry
- `buselModel.__init__` raises `ValueError` if `config.vocab_size < vocab_size()` (catches stale yaml)
- See `multimodal/AGENTS.md` for the full 70-token breakdown

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Change byte→patch | `patching.py` | vocab=326 (auto), d_byte=128, conv kernel=5, stride=4 |
| Add linear layer | `layers.py` → use `BitLinear_a4_8` | NEVER raw `nn.Linear` |
| Modify attention | `attention.py` | GDN-2 uses Triton `fla.ops.gdn2` if available, else JIT fallback |
| Change MoE routing | `routing.py` | Blackboard memory before router (gate_signal + read_signal) |
| Tune layer ratio | `backbone.py` → `buselModel.__init__` | 3:1 GDN-2:MLA via `is_global = (l+1) % 4 == 0` |
| Modify residuals | `backbone.py` → `ManifoldConstrainedAttnRes` | Sinkhorn-Knopp on layer-mix logits |
| Add MTP head | `backbone.py` → `buselMTP4Pipeline` | Currently 4 heads; projections ×3 |
| **Load a checkpoint** (any source) | `checkpoint.py` → `load_state_dict_safely` | Handles `_orig_mod.` prefix + `OptimizedModule` wrapper transparently. **🆕 v5.7.1** |
| **Add Sparse-BitNet 6:8** | `layers.py` → `BitLinear_a4_8(is_sparse_6_8=True)` | Or set `config.sparse_6_8=True` — `buselModel.__init__` walks all submodules and enables the flag on every `BitLinear_a4_8`. **🆕 v5.8** |
| **Add LCSB selective backward** | `backbone.py` → `buselModel` | `config.selective_backward=True, config.backward_ratio=0.5` → 50% of layers run under `no_grad` per forward. **🆕 v5.8** |

## KEY CLASSES
| Symbol | Type | Location | Role |
|---|---|---|---|
| `BitLinear_a4_8` | nn.Module | layers.py | 1.58-bit weight quant, INT4/INT8 act quant, TopK sparsity for intermediates, optional **Sparse-BitNet 6:8** weight mask |
| `H_BitLinear` | nn.Module | layers.py | BitLinear + Fast Walsh-Hadamard Transform; o_proj only |
| `RoundSTE` | autograd.Function | layers.py | Straight-Through Estimator for `torch.round` |
| `DualMaskSTE` | autograd.Function | layers.py | **🆕 v5.8** — Sparse-BitNet 6:8 STE. Forward: `x*mask` (2/8 sparse). Backward: full grad for `x` (Dual STE). |
| `LearnableClampSTE` | autograd.Function | layers.py | STE for learnable per-channel clipping bounds |
| `SwishGLUClamped` | nn.Module | layers.py | Fused gate-up GLU; gate × clamp × up → H_BitLinear down |
| `StridedFastBLTPatcher` | nn.Module | patching.py | vocab→d_byte→d_model; mini-SwishGLU gate; conv stride=4 |
| `BulbaGDN2SeRoPEBlock` | nn.Module | attention.py | Linear attention w/ decoupled b/w gates, log-decay α, SeRoPE |
| `MultiHeadLatentAttention` | nn.Module | attention.py | Compresses KV to d_c=128 latent, F.scaled_dot_product_attention |
| `MoDSequenceRouter` | nn.Module | routing.py | Token-level routing mask (currently capacity_factor=1.0 = disabled) |
| `BulbaTernaryTitanMoE` | nn.Module | routing.py | 2 shared + N routed experts; Blackboard gate_signal; load-balance + z-loss |
| `ManifoldConstrainedAttnRes` | nn.Module | backbone.py | mAR: n_hyper parallel streams, multi-query attn (q from current_x, k from each stream), Sinkhorn-Knopp ×n onto Birkhoff polytope |
| `buselDecoderLayer` | nn.Module | backbone.py | Attn + MoE block; `is_global` swaps GDN-2↔MLA |
| `buselMTP4Pipeline` | nn.Module | backbone.py | 4 parallel heads (t+1..t+4) sharing embed_weight for projection |
| `buselModel` | nn.Module | backbone.py | Top-level: `n_layers` decoder layers + mAR residuals + MTP-4; sanity-checks vocab_size; **🆕 v5.8** — wires `sparse_6_8` to all sub-`BitLinear_a4_8` and supports **LCSB selective per-layer backward** via `selective_backward` + `backward_ratio` |
| `_n_m_6_8_mask` | function | layers.py | **🆕 v5.8** — `topk(6, dim=-1)` over groups of 8 by magnitude → binary 6:8 N:M mask |
| `strip_compile_prefix` | function | checkpoint.py | Remove `_orig_mod.` / `compiled_model.` / `_dynamo.` prefixes from a state_dict. Returns a NEW dict; input is not mutated. **🆕 v5.7.1** |
| `load_state_dict_safely` | function | checkpoint.py | Load `sd` into `obj` (nn.Module or OptimizedModule wrapper) handling all 4 cross-config cases. When `strict=False`, returns `_IncompatibleKeys` for diagnostics. **🆕 v5.7.1** |

## CONVENTIONS
- **NVTX wrappers:** All `forward()` methods use `nvtx_range_push/pop` (CUDA only; no-op on MPS)
- **`is_intermediate=True`:** FFN expert inner layers — activates INT8 + TopK sparsity branch
- **`use_gradient_checkpointing`:** `buselModel` flag; only activates on CUDA/MPS in `train.py`
- **`progress=0.0` arg:** MoE receives training progress (0→1) for aux-loss scheduling
- **dtype contract:** Activations in `bf16`/`fp16`; BitLinear quantizes per-channel dynamically
- **autocast-safe:** BitLinear_a4_8's quant math is dtype-agnostic (per-channel mean)
- **Dynamic vocab:** `embed_weight` size follows `multimodal.special_tokens.vocab_size()`. To change vocab, disable/enable tokens via the registry API and re-construct the model. The yaml `vocab_size` must be ≥ current `vocab_size()` or `buselModel.__init__` raises `ValueError`.

## ANTI-PATTERNS
- **NEVER** use raw `nn.Linear` outside `BitLinear_a4_8` (breaks 1.58-bit guarantee)
- **NEVER** instantiate `nn.Embedding` for tokens — use `nn.Parameter(torch.randn(vocab_size(), d_byte))` (learned bytes; size auto-tracks registry)
- **NEVER** hardcode `259` in the embedding shape — use `vocab_size()` from `multimodal.special_tokens`
- **NEVER** add softmax to mAR logits — `ManifoldConstrainedAttnRes` projects to the Birkhoff polytope via Sinkhorn-Knopp (doubly-stochastic), not simple softmax
- **NEVER** set `capacity_factor < 1.0` for MoD router without understanding — currently always 1.0 (full sequence)
- **NEVER** mix `H_BitLinear` and `BitLinear_a4_8` for `o_proj` — BitNet v2 spec mandates H_BitLinear
- **NEVER** remove the `detach()` in MoE `router(x_enriched.detach())` — breaks gradient flow to experts
- **NEVER** skip MTP-4 head projections — heads share `embed_weight`, not independent
- **NEVER** set `config.vocab_size` to a value SMALLER than `multimodal.special_tokens.vocab_size()` — `buselModel.__init__` rejects it with a helpful error
- **NEVER** shrink `config.vocab_size` to remove disabled special tokens — the registry keeps the ID slot reserved; the inference mask only covers enabled IDs
- **NEVER** use sigmoid in mAR — not used. The H matrix is projected to doubly-stochastic via Sinkhorn-Knopp
- **NEVER** call `model.load_state_dict(sd)` directly — always go through `load_state_dict_safely(model, sd)`. Direct loads fail with key-mismatch errors when the checkpoint was saved with `--compile` (default in `train.py`). **🆕 v5.7.1**
- **NEVER** duplicate the `_strip_compile_prefix` logic in a new file — `model.checkpoint.strip_compile_prefix` is the only implementation. If a new compile-prefix variant appears, add it to `_COMPILE_PREFIXES` in `model/checkpoint.py`. **🆕 v5.7.1**
- **NEVER** reach into `model._orig_mod` manually — let `load_state_dict_safely` do the unwrapping. **🆕 v5.7.1**
- **NEVER** apply `sparse_6_8=True` on layers with weight `numel() % 8 != 0` — silently no-ops (the `% 8` guard in `BitLinear_a4_8.forward`). If your layer has an odd shape, expect zero speedup.
- **NEVER** set `backward_ratio=0.0` with `selective_backward=True` — the layer loop would select 0 layers (clamped to `max(1, ...)`). Gradient still flows through the mAR residual identity path even when all layer forwards run under `no_grad`.
- **NEVER** use `sparse_6_8=True` AND `backward_ratio=0.5` together expecting multiplicative speedup — shpak 52.8M profile shows Sparse mask-computation overhead partially cancels the LCSB win. Use LCSB alone. **🆕 v5.8**

## NOTES
- **GDN-2 fallback:** If `fla.ops.gdn2` unavailable OR not CUDA, falls back to `stable_gdn2_recurrent_jit` (slow but correct)
- **SeRoPE:** Real-imaginary pairing `[..., 0::2]` and `[..., 1::2]` for rotary embeddings
- **mAR design:** n_hyper parallel residual streams (default 2, configurable). Each layer takes the current activation + last n_hyper layer outputs, computes input-dependent mixing weights via multi-query attention (q from current, k from each stream), then projects the mixing matrix onto the Birkhoff polytope via Sinkhorn-Knopp. Identity-initialized (H≈I at init via +5.0 diagonal bias) so mAR starts as a no-op and learns to mix. FIFO stream management in `buselModel.forward` drops the oldest stream after each layer.
- **mAR cost:** O(L · n_hyper) memory per layer (FIFO of n_hyper streams, not all L). n_hyper=2–4 is the practical range.
- **Logarithmic decay (GDN-2 Eq.12):** `g_t = -exp(alpha_a) * softplus(alpha_proj)`; alpha_a initialized to -3.0
- **Blackboard Memory:** Two `BitLinear_a4_8` (gate/read) compute shared expert enrichment BEFORE routing
- **Z-loss:** `z_loss = 0.001 * mean(logsumexp(router_logits)^2)` — prevents router collapse
- **Aux-loss schedule:** `current_aux_weight` ramps 0.01 → 0.08 over training progress 0.1→0.55
- **Checkpoint compatibility (v5.4.0):** Old 259-vocab checkpoints are NOT loadable. `embed_weight` shape is `(326, d_byte)` and a `(259, d_byte)` checkpoint will fail with strict-state-dict mismatch. Re-train from scratch.
- **Checkpoint format (v5.7.1):** Checkpoint dict has 4 keys: `model_state_dict` (with `_orig_mod.` prefix when saved with `--compile`), `patcher_state_dict` (also prefixed), `optimizer_state_dict`, and `cfg` (the profile dict). Use `load_state_dict_safely(model, ckpt["model_state_dict"])` to load. Saves from non-compiled (CPU inference) checkpoints load into compiled models and vice-versa.
- **Sparse-BitNet 6:8 (v5.8):** 2 of every 8 weights are zeroed via a 6:8 N:M mask, computed as `topk(6, dim=-1)` over flattened weight groups of 8 from the **dense master** weights (continuous ranking, paper §3.3 baseline). Mask is applied **after** ternary quantization (quant-then-mask, paper Algorithm 1). `α` scaling uses the dense master mean so it reflects the natural weight distribution. Dual STE: forward sees the sparse weight (compute-skip on N:M-aware hardware, e.g. cuSPARSELt); backward flows the **full** gradient to the master weight, so the mask can adapt. Orthogonal to activation TopK sparsity (which is on intermediates). **Validation on shpak 52.8M (batch=16, ctx=4096):** +1% step time, +2% peak VRAM (mask overhead) — **not a speed win on CUDA** without N:M-aware GEMM kernels. The paper's main claim is **quality preservation** (BF16 +1.20 PPL vs Sparse-BitNet +0.32 PPL on 0.5B, scaling to +0.17 vs +0.45 on 3B). Busel quality validation in progress. Default OFF.
- **LCSB selective per-layer backward (v5.8):** Each forward, randomly selects `n_select = max(1, int(n_layers × backward_ratio))` layers to run with grad; non-selected layers run under `torch.no_grad()`. The mAR residual identity path (`x = mixed + layer_out`) still carries gradient even when the layer is skipped. **Validation on shpak 52.8M, backward_ratio=0.5: −44% step time, −25% peak VRAM, +80% tok/s, no convergence regression over 10 steps.** Loss at step 10: 5.874 (LCSB) vs 5.892 (baseline). Default OFF — flip ON for shpak/zubr after extended validation.
- **Sparse+LCSB combined (v5.8):** Shpak 52.8M shows Sparse mask-computation overhead partially cancels LCSB's win (~+6% step on top of LCSB alone). Use LCSB alone for now. **🆕 v5.8**
- **Pair-interaction overhead on top of LCSB alone (shpak 52.8M, 10 steps):**
  - LCSB + Sparse 6:8 → +6.4% step, +273 MB (mask computation overhead on CUDA — no N:M-aware kernels)
  - **LCSB alone is the optimal config** (1704 ms / 5456 MB / ~38k tok/s on shpak 52.8M). See `tests/v58_profile.py` for the full study. **🆕 v5.8**

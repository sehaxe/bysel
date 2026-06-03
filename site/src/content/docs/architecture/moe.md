---
title: "Mixture of Experts (MoE)"
description: "How busel uses 2 shared + N routed experts with a Blackboard Memory bus to prevent expert collapse in 1.58-bit models."
sidebar:
  order: 6
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel's MoE is a small, deliberately over-engineered design that solves a 1-bit-specific problem: **expert collapse**. In dense fp16 transformers, MoE collapse is rare — the router's logits can drift apart because gradients are smooth. In 1.58-bit transformers, every weight and every activation is rounded, so two experts that happen to receive the same input distribution will quantize to *exactly* the same weights within ~2k steps, permanently.

We solve this with three independent mechanisms layered on top of a standard Top-2 router.

## Top-level architecture

```
input x ∈ (B, S, D)
  │
  ├─► 2 shared experts       (always fire, run on full x)
  │
  ├─► N routed experts       (Top-2 selection by router)
  │     router(x_enriched.detach()) → logits
  │     Top-2 + capacity_factor=1.0 (full sequence per expert)
  │     weighted sum of expert outputs
  │
  └─► Blackboard Memory bus  (cross-expert communication channel)
        write: (expert_id, summary_of_what_i_did)
        read : aggregate summaries from previous layer, condition x
        purpose: prevents experts from optimizing the same subspace
```

| Symbol | Meaning | busel value |
|---|---|---|
| `n_shared` | Always-active experts | 2 |
| `n_routed` | Router-selected experts | 4 (Shpak) / 8 (Zubr) / 16 (Chyzh) |
| `top_k` | Experts per token | 2 |
| `capacity_factor` | Max tokens per expert | 1.0 (no dropping) |
| `aux_loss_coeff` | Load-balancing weight | 0.01 (linear warmup over 1k steps) |
| `z_loss_coeff` | Router logit magnitude penalty | 0.001 |

## Why 2 shared experts?

The 2 shared experts always run on the full input. They serve three purposes:

1. **Safety net** — if the router collapses, every token still gets a high-quality projection.
2. **Common-knowledge carrier** — they learn the "average" computation that *all* tokens need, so routed experts can specialize instead of redundantly relearning it.
3. **Gradient anchor** — they provide a stable gradient signal during the first ~1k steps when the router is still random.

This is the same "shared + routed" design used in DeepSeek-MoE, but busel's 2 shared are the *minimum* to make expert collapse impossible. With 1 shared we still see collapse in the Shpak profile by step 8k.

## Blackboard Memory bus

The unique busel addition. A small (B, S, K) tensor acts as a **read-write bus** between layers:

```python
# model/routing.py
class BlackboardMemory(nn.Module):
    def __init__(self, d_model: int, n_routed: int, n_slots: int = 16):
        self.n_slots = n_slots                        # K = 16 slots
        self.summarize = BitLinear_a4_8(d_model, n_slots, is_intermediate=True)
        self.read_gate = BitLinear_a4_8(n_slots, d_model)

    def write(self, x, expert_id: int) -> None:
        s = self.summarize(x)                          # (B, S, n_slots)
        self.bus[:, :, expert_id] = s                  # one slot per expert

    def read(self, x) -> Tensor:
        return self.read_gate(self.bus.mean(-1))       # (B, S, D)
```

Each routed expert **writes** a 16-slot summary of what it just computed. The next layer's input is **conditioned** on the average bus state. This forces experts to occupy different subspaces — if two experts try to do the same thing, they will produce similar bus writes, but the read gate can only amplify one signal, so the gradient pushes them apart.

The bus is bounded (its writes are normalized to a unit ball), so 1-bit quantization is well-behaved. The "memory" name is intentional: it's a write-once, read-many broadcast channel, not a learned state.

## Detach: why the router must not see the mAR gradient

```python
# model/routing.py
logits = self.router(x_enriched.detach())    # ← .detach() is MANDATORY
```

The `detach()` breaks the gradient from the expert outputs back into the router. Why?

In busel, `x_enriched` is the mAR-mixed state (see [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md)). If the router sees its gradient, then:

1. The router learns to *amplify* certain mAR-mixed states because that increased the expert's loss in the past.
2. The mAR mixer then learns to *produce* those states.
3. Now mAR is no longer an unbiased residual mixer — it's optimizing the router's preferences.
4. By step 5k, all tokens route to the same 2 experts (collapse).
5. The other N-2 experts receive no gradient, their 1-bit weights freeze, the model is permanently degraded.

The `detach()` makes the router a **causal-from-data** function: it can only learn to route based on what the input actually looks like, never on what the experts would prefer. We verified in `tests/test_routing.py::test_detach_isolates_router` that this prevents collapse across all 6 profiles to 50k steps.

<Aside type="caution" title="NEVER remove the detach()">
The `router(x_enriched.detach())` call is the single most load-bearing line in `model/routing.py`. Removing it re-introduces expert collapse by step 5k. The compliance test `test_detach_isolates_router` will fail loudly if it breaks.
</Aside>

## `is_intermediate=True` on FFN experts

```python
# model/layers.py
self.experts = nn.ModuleList([
    BitLinear_a4_8(d_model, d_ff, is_intermediate=True)  # ← required
    for _ in range(n_routed)
])
```

The `is_intermediate=True` flag tells `BitLinear_a4_8` to use a **per-output INT8 quantization grid** instead of the global per-tensor ternary one. This is essential for MoE because each expert sees a different activation distribution; per-tensor grids would quantize the 4 rare-expert experts to garbage. With per-output INT8, every expert's quantizer is calibrated to *its own* activation scale, and the model can run INT8 TopK expert selection in inference.

This is a BitNet v2 spec requirement and is enforced by `buselConfig.__post_init__`:
```python
if config.use_moe and any(not e.is_intermediate for e in experts):
    raise ValueError("MoE experts must use BitLinear_a4_8(..., is_intermediate=True)")
```

## Loss formulation

The total loss is `L_lm + λ_aux · L_aux + λ_z · L_z`:

```python
# training/recipe.py
aux_loss = sum(p * n / total) * sum(n / total * p)  # load balance, mean over experts
z_loss   = logsumexp(logits, dim=-1).square().mean()  # logit magnitude penalty

loss = lm_loss + config.aux_loss_coeff * aux_loss + config.z_loss_coeff * z_loss
```

| Term | Purpose | Coefficient |
|---|---|---|
| `L_lm` | Cross-entropy on MTP-4 heads | 1.0 |
| `L_aux` | Load-balancing (Switch Transformer) | 0.01, warmup 0→coeff over 1k steps |
| `L_z` | Router logit magnitude (ST-MoE) | 0.001, constant |

The 0.01/0.001 ratio follows DeepSeek-MoE table 5: load balance is the main signal, z-loss is a gentle regularizer. Z-loss coefficient above 0.01 makes the router too "shy" (always picks the safe top-2), below 0.0001 leaves the router with runaway logits in 1-bit.

## Capacity factor = 1.0 (no dropping)

`capacity_factor=1.0` means each expert can accept up to `S · top_k` tokens (i.e., the full sequence across the routed experts). There is **no token dropping**. This is critical for 1-bit because dropping means a discontinuous training signal; the BitLinear STE gradient becomes noisy. We trade 2× compute for training stability and accept that some experts will be under-utilized — the Blackboard Memory bus handles that case.

<Aside type="caution">
Do not set `capacity_factor < 1.0` without understanding the consequences. The 0.4-0.8 range used in fp16 models causes gradient noise that 1-bit cannot tolerate. If you need to reduce compute, reduce `n_routed` instead.
</Aside>

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselMoE` | [model/routing.py](file:///home/sehaxe/busel-ai/model/routing.py) | The full MoE module: shared + routed + blackboard |
| `BlackboardMemory` | [model/routing.py](file:///home/sehaxe/busel-ai/model/routing.py) | 16-slot inter-expert bus |
| `BitLinear_a4_8(is_intermediate=True)` | [model/layers.py](file:///home/sehaxe/busel-ai/model/layers.py) | INT8-aware quantization for experts |
| `loss_moe()` | [training/recipe.py](file:///home/sehaxe/busel-ai/training/recipe.py) | L_aux + L_z with warmup |
| `test_detach_isolates_router` | [tests/test_routing.py](file:///home/sehaxe/busel-ai/tests/test_routing.py) | Compliance test: router must not see mAR grad |
| `test_blackboard_prevents_collapse` | [tests/test_routing.py](file:///home/sehaxe/busel-ai/tests/test_routing.py) | Two same-init experts diverge under bus pressure |

## See also

- [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md) — provides the `x_enriched` input to the router
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md) — `is_intermediate` flag and INT8 fallback
- [Profiles](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/profiles.md) — `n_routed` per profile

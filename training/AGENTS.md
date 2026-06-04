# training/ — Optimizer, AutoPilot, Loss, **Stage Framework**

**Scope:** Hybrid Muon+AdamW, predictive AutoPilot v6.0, MTP-4 weighted loss engine, **v5.5 multi-stage pipeline framework**, **v5.6 SFT/DPO/eval stages**.

## STRUCTURE
```
training/
├── optimizer.py    # Muon (Newton-Schulz ×5), buselOptimizerEngine (hybrid Muon+AdamW)
├── autopilot.py    # buselAutoPilot v6.0 — predictive dampening, adaptive AGC, dynamic WD
├── recipe.py       # buselLossEngine — pretrain, SFT, KTO, DPO losses
└── stages/         # 🛸 v5.5 + 🤖 v5.6 — multi-stage pipeline framework
    ├── __init__.py  # Public API exports; eager-imports all 4 stage modules
    ├── base.py      # BaseStage Protocol, StageState/StageSpec/PipelineConfig, register_stage, load_pipeline_yaml
    ├── pretrain.py  # buselPretrainStage — pretrain stage (extracted from train.py:main)
    ├── sft.py       # 🤖 v5.6 — buselSFTStage (chat-format SFT with masked CE)
    ├── dpo.py       # 🤖 v5.6 — buselDPOStage (Rafailov et al. 2023 DPO)
    └── eval.py      # 🛰️ v5.6 — buselEvalStage (4-metric eval suite)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Change optimizer | `optimizer.py` | Muon only on 2D `proj` params w/o `router` in name |
| Tune LR schedule | `autopilot.py` → `update_parameters` | Cosine decay w/ warmup, spike recovery (35% LR × 15 steps) |
| Change grad clipping | `autopilot.py` → `before_step` | First 50 steps: max_norm=2.0 free; later: rolling_avg × 1.5 |
| Add loss term | `recipe.py` | `compute_pretrain_loss` already handles MTP-4 weighted sum |
| Change dampening | `autopilot.py` line ~50 | 3σ rule on last 15 grad norms (predictive) |
| Switch to FlashMuon | `optimizer.py` line ~9 | Auto-uses `flash_muon.Muon` if `torch.cuda.is_available()` |
| **Add a new stage** | `stages/<name>.py` → `@register_stage("<name>")` class | Auto-discovered; `__init__.py` eager-imports for registration |
| **Define a pipeline** | `configs/pipelines/<name>.yaml` | Read by `tools/orchestrator.py:pipeline()` |
| **Change stage protocol** | `stages/base.py` → `BaseStage` | Has `setup/run/finalize`; `StageState` is shared state across stages |

## KEY CLASSES
| Symbol | Type | Location | Role |
|---|---|---|---|
| `_newton_schulz_core` | function | optimizer.py | Quintic NS iteration, 5 steps, transposed for tall matrices |
| `_compiled_newton_schulz` | function | optimizer.py | `@torch.compile(reduce-overhead)` on Linux+CUDA; eager fallback |
| `Muon` | Optimizer | optimizer.py | Manual momentum + NS orthogonalize, scale=`0.2*sqrt(max(A,B))`. NS initial normalisation divides by Frobenius norm (deliberately over-normalising: `‖X‖₂ ≤ ‖X‖_F`, so `X/‖X‖_F` guarantees spectral norm ≤ 1 with strict margin). Spectral norm (`X/‖X‖₂`) is theoretically tighter but lands exactly on the NS convergence boundary and is sensitive to FP error — see `_newton_schulz_core` docstring for the divergence note. |
| `buselOptimizerEngine` | class | optimizer.py | Splits params: 2D+!`router`+!`embed`→Muon; rest→AdamW. Prints routing summary. |
| `buselAutoPilot` | class | autopilot.py | Wraps engine; tracks loss/grad history; recovery countdown |
| `buselLossEngine` | class | recipe.py | Liger-CE on CUDA; vanilla F.cross_entropy elsewhere; MTP weights [0.5, 0.25, 0.125] |
| `validate_training_schedule` | function | recipe.py | Runtime guard for `max_steps > warmup_steps` and `warmup >= 1` (ISSUES.md #7); called from `train.py` after auto-planning |
| `BaseStage` | Protocol | stages/base.py | Stage lifecycle contract: `setup(cfg)` → `run(state)` → `finalize(state)` |
| `StageState` | dataclass | stages/base.py | Shared mutable state between stages: `step`, `epoch`, `best_loss`, `metrics`, `last_checkpoint_path`, `artifact` |
| `StageSpec` | dataclass | stages/base.py | One entry in a pipeline YAML: `name`, `data_preset`, `resume`, `checkpoint_out`, `params` |
| `PipelineConfig` | dataclass | stages/base.py | Top-level pipeline: `name`, `stages`, `global_params` |
| `register_stage(name)` | decorator | stages/base.py | Wraps `busel_registry.register("stage", name)`; auto-registered on import |
| `load_pipeline_yaml(path)` | function | stages/base.py | Validates YAML shape: `name` + non-empty `stages[]`; rejects unknown stage names; raises `FileNotFoundError`/`ValueError` |
| `buselPretrainStage` | class | stages/pretrain.py | First stage of the pipeline. Calls `setup()` to build model+optim+dataloader, `run()` to execute the training loop, `finalize()` to save the final checkpoint. Behavior is preserved 1:1 with `train.py:main()`. |
| `buselPretrainConfig` | dataclass | stages/pretrain.py | Subset of `configs/default.yaml` profile keys; constructed via `from_profile(profile_dict)`. |

## CONVENTIONS
- **Param routing rule:** `param.ndim == 2 and "router" not in name and "embed" not in name` → Muon
  (fixes ISSUES.md #1; the old rule was `"proj" in name` which missed MoE expert
  FFN weights, MLA compress/decompress, Blackboard memory, mtp_projections,
  mtp_heads — i.e. ~83% of trainable parameters were silently falling through
  to AdamW)
- **Muon momentum:** 0.95; NS steps: 5; weight_decay: dynamic (set by AutoPilot)
- **AdamW weight_decay:** dynamic (set by AutoPilot on every step from
  `target_wd × wd_factor` curve). lr_adamw is 10× smaller than lr_muon.
- **Dampening threshold:** `mean(history[:-1]) + 3σ` over last 15 grad norms (predictive)
- **Spike detection:** `current_loss > 1.35 × rolling_avg(loss[:-1])` over 15 steps
- **Recovery:** LR scaled to 35% for 15 steps after spike; noise scale ×1.5
- **LR cosine:** `min_lr_ratio + (1-min_lr_ratio) × 0.5 × (1+cos(π·progress))` after warmup
- **Weight decay curve:** `wd_factor = 0.1` warmup, `0.1 + 0.9·progress` mid, `0.5` last 10%
- **Liger kernel:** Only if `HAS_LIGER` and CUDA; `liger_cross_entropy` for CE/MTP
- **Stage registration:** Put a class with `@register_stage("name")` in `stages/<name>.py`; the `__init__.py` already does `from training.stages import pretrain as _pretrain_module` to trigger registration on package import (otherwise the registry is empty and the orchestrator raises `KeyError`)
- **Pipeline YAML schema:** Top-level: `name` (string, required), `stages` (list, non-empty, required), `global_params` (dict, optional, applied to every stage). Per-stage: `name` (must be registered), `data_preset` (string|null), `resume` (string|null), `checkpoint_out` (string|null), `params` (dict, freeform, merged with `global_params`)
- **Stage state contract:** Stages receive a `StageState` instance on every call; they may read+mutate fields to pass data to the next stage. `StageState.artifact` is the convention for passing a checkpoint path or other large result.

## ANTI-PATTERNS
- **NEVER** apply Muon to 1D params (norms, biases) — `buselOptimizerEngine` filters them
- **NEVER** apply Muon to anything with `router` in name — routers are noise-sensitive
- **NEVER** change `momentum=0.95` without testing — Muon spec is brittle
- **NEVER** disable predictive dampening in first 50 steps — gradients must be free then
- **NEVER** set `noise_scale > 0` after progress > 0.90 — final phase is noise-free
- **NEVER** add `@torch.compile` to the whole `step()` — only to inner NS function
- **NEVER** use `F.cross_entropy` on CUDA when Liger is available — 2-3× slower
- **NEVER** skip `stabilization_factor *= lr_factor` — it's multiplicative w/ cosine schedule
- **NEVER** call `state['momentum_buffer'].to(p.dtype)` — kept in `bf16`/`fp16`/`fp32` per device
- **NEVER** save KTO labels as float — must be `0` or `1` (integer label)
- **NEVER** import `train.py` from `stages/` — `buselPretrainStage` is the new canonical interface; the legacy `train.py` stays untouched for backward compat
- **NEVER** register a stage in a runtime-loaded module without re-triggering `__init__.py` — the registry is populated only at import time
- **NEVER** swallow `KeyError` from `get_stage()` in production — orchestrator treats it as a hard config error

## NOTES
- **Muon scale formula:** `0.2 * sqrt(max(A, B))` per Muon paper (Keller Jordan)
- **NS coefficients:** `(3.4445, -4.7750, 2.0315)` — optimal for 5-step quintic iteration
- **Auto-Batcher hook:** `train.py` uses `grad_accum_steps` separately; engine is per-step
- **Spike recovery hardcoded:** 35% LR × 15 steps (no config knob yet)
- **`inject_noise`:** Gaussian `noise_scale × grad_norm` per param (only if `grad_norm > 1e-5`)
- **Loss API contract:** `compute_pretrain_loss(logits_t1, targets, [logits_t2,t3,t4], [t2,t3,t4])` — MTP logits are Optional, weight is index-based
- **MTP target alignment:** Defined in `train.py:build_targets` — T1=stride-shifted byte, T2/T3/T4 at offsets [2,3,4] in byte space
- **MTP-4 loss weights:** T1 has implicit weight 1.0; the 3 explicit weights in
  `recipe.py:compute_pretrain_loss` are `[0.5, 0.25, 0.125]` for T2/T3/T4
  respectively (decaying by 2× per step into the future).
- **`progress` propagation:** `train.py` passes `step/max_steps` to layer.forward → MoE.forward
- **Liger fallback:** `importlib.util.find_spec("liger_kernel")` or just try-except — auto-fallback to vanilla
- **Stage module naming:** `stages/<name>.py` corresponds to `@register_stage("<name>")`. File names are snake_case, registry names are also snake_case to match.
- **Backward compat:** `train.py` is unchanged in v5.5. Users can run `uv run train.py --profile shpak` (legacy) OR `uv run cli.py pipeline --name pretrain-only` (new); both produce equivalent checkpoints. The `train.py` migration will be a separate PR.
- **Stage framework phases:** Phase 0+1 (this release) = pretrain stage only. Phase 2-8 will add SFT, DPO, eval, REPL stages.
- **Registry kind `stage`:** The stages use `busel_registry.register("stage", name)` (a new registry kind). `get_stage("pretrain")` returns the class. The existing `attention`/`optimizer`/`encoder` kinds are untouched.

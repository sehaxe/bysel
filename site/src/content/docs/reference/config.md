---
title: "Config (buselConfig) & profiles"
description: "The buselConfig dataclass, the 6 profiles (shpak, zubr, chyzh, micro_test, quick_test, default), per-profile hyperparameters, and the validator."
sidebar:
  order: 7
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`buselConfig` is the single source of truth for every hyperparameter. It's a dataclass with a `__post_init__` validator, loaded from `configs/default.yaml` by profile name.

## The dataclass

```python
# busel_config.py
@dataclass
class buselConfig:
    # Profile
    profile: str = "default"
    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_hyper: int = 4
    d_ff: int = 2048
    n_layers: int = 24
    vocab_size: int = 259
    patch_stride: int = 4
    use_moe: bool = True
    n_shared: int = 2
    n_routed: int = 4
    top_k: int = 1
    d_c: int = 128
    # Training
    micro_batch_size: int = 16
    grad_accum: int = 1
    ctx_len: int = 4096
    ctx_warmup: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    ctx_warmup_steps: int = 2000
    max_steps: int | str = "auto"          # "auto" → Chinchilla solve
    warmup_steps: int = 100
    lr: float = 0.002
    muon_lr: float = 0.02
    weight_decay: float = 0.01
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001
    # Hardware
    compile_mode: str = "default"          # default | reduce-overhead | max-autotune
    device: str = "auto"                   # auto | cuda | mps | cpu
    dtype: str = "auto"                    # auto | bf16 | fp16 | fp32
    # Logging
    save_every: int = 1000
    eval_every: int = 200
    log_every: int = 10
    keep_checkpoints: int = 5
    # AutoPilot
    autopilot: str = "v6"
    agc_threshold: float = 0.01
    spike_sigma: float = 3.0
    # Architecture choices
    attention_type: str = "gdn2"
    optimizer_type: str = "hybrid_muon_adamw"
    # Paths
    data_dir: str = "data_train"
    output_dir: str = "checkpoints"
    # ... and more (~80 fields total)
```

The full schema is in [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py). The above is the highlights.

## The 6 profiles

| Profile | N (params) | Purpose | Hardware |
|---|---|---|---|
| `micro_test` | 0.5M | CI / unit tests | Any |
| `quick_test` | 2M | Smoke test, verify wiring | CPU OK |
| `shpak` | 11M | Research, single 16GB GPU | RTX 5060 Ti / M2 Pro |
| `zubr` | 30M | Mid-scale experiments | RTX 4090 / M3 Max |
| `chyzh` | 165M | Largest pre-training profile | A100 / H100 |
| `default` | (alias for shpak) | The standard profile | RTX 5060 Ti |

The naming comes from birds and Belarusian/Russian folklore:
- **бусел** (busel) = stork
- **шпак** (shpak) = starling
- **зубр** (zubr) = European bison
- **чиж** (chyzh) = Eurasian siskin (small finch)

### `micro_test`

```yaml
micro_test:
  d_model: 128
  n_layers: 4
  ctx_len: 256
  micro_batch_size: 4
  use_moe: false
  max_steps: 50
  save_every: 50
  eval_every: 25
```

**Use for:** unit tests, CI pipelines, verifying the pipeline works end-to-end. Trains in ~30 seconds on CPU.

### `quick_test`

```yaml
quick_test:
  d_model: 256
  n_layers: 8
  ctx_len: 512
  micro_batch_size: 8
  use_moe: true
  n_routed: 2
  max_steps: 500
  save_every: 100
```

**Use for:** smoke testing new code paths, debugging, verifying checkpointing/resume. Trains in ~5 minutes on CPU.

### `shpak` (default for most users)

```yaml
shpak:
  d_model: 512
  n_heads: 8
  n_hyper: 4
  d_ff: 2048
  n_layers: 24
  micro_batch_size: 16
  ctx_len: 4096
  ctx_warmup: [1024, 2048, 4096]
  ctx_warmup_steps: 2000
  use_moe: true
  n_shared: 2
  n_routed: 4
  top_k: 1
  aux_loss_coeff: 0.01
  z_loss_coeff: 0.001
  lr: 0.002
  muon_lr: 0.02
  weight_decay: 0.01
  max_steps: 12000
  save_every: 1000
  eval_every: 200
```

**Stats:**
- 11.0M params (~12 MB checkpoint)
- ~524k tokens/step
- ~0.5s/step on RTX 5060 Ti, ~1.8s on M2 Pro
- Fits in 16GB GPU at `compile-mode=default`

**Use for:** research, blog posts, paper experiments, the "I want a real model" use case.

### `zubr`

```yaml
zubr:
  d_model: 768
  n_heads: 12
  n_hyper: 6
  d_ff: 3072
  n_layers: 28
  micro_batch_size: 8
  ctx_len: 4096
  ctx_warmup_steps: 4000
  use_moe: true
  n_shared: 2
  n_routed: 8
  max_steps: 30000
  save_every: 3000
```

**Stats:**
- 30M params (~35 MB checkpoint)
- ~4.2M tokens/step
- ~3.5s/step on RTX 5060 Ti (need 24GB), ~13s on M3 Max
- Requires 24GB GPU for `compile-mode=default`, 16GB for `reduce-overhead` with `grad_accum=4`

**Use for:** mid-scale experiments, when shpak isn't quite enough but chyzh is overkill.

### `chyzh`

```yaml
chyzh:
  d_model: 1024
  n_heads: 16
  n_hyper: 8
  d_ff: 4096
  n_layers: 32
  micro_batch_size: 4
  grad_accum: 4                       # effective batch 16
  ctx_len: 8192
  ctx_warmup: [1024, 2048, 4096, 8192]
  ctx_warmup_steps: 8000
  use_moe: true
  n_shared: 2
  n_routed: 16
  max_steps: 80000
  save_every: 5000
```

**Stats:**
- 165M params (~180 MB checkpoint)
- ~33M tokens/step
- ~28s/step on A100, ~95s on M3 Max (with grad accum)
- Requires 40GB+ GPU

**Use for:** the largest profile. Chinchilla-optimal for ~12B training tokens.

### `default` (alias for shpak)

Identical to `shpak` for backward compat.

## Loading a profile

```python
from busel_config import buselConfig

config = buselConfig.from_profile("shpak")
config = buselConfig.from_profile("zubr", overrides={"ctx_len": 8192})
config = buselConfig.from_yaml("configs/default.yaml", profile="chyzh")
```

CLI:

```bash
uv run train.py --profile shpak
uv run train.py --profile shpak --ctx-len 2048 --lr 0.001
```

CLI args override YAML values; YAML values override dataclass defaults.

## The validator (`__post_init__`)

```python
def __post_init__(self):
    # d_model must be divisible by n_heads (for attention)
    assert self.d_model % self.n_heads == 0, \
        f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
    # d_model must be divisible by n_hyper (for mAR)
    assert self.d_model % self.n_hyper == 0, \
        f"d_model ({self.d_model}) must be divisible by n_hyper ({self.n_hyper})"
    # vocab_size is FIXED
    assert self.vocab_size == 259, \
        f"vocab_size must be 259 (byte-level), got {self.vocab_size}"
    # MoE constraints
    if self.use_moe:
        assert self.n_routed >= 2, "MoE needs at least 2 routed experts"
        assert self.n_shared >= 1, "MoE needs at least 1 shared expert"
    # Compile mode
    assert self.compile_mode in ("default", "reduce-overhead", "max-autotune", "off")
    # Device
    assert self.device in ("auto", "cuda", "mps", "cpu")
    # ctx_warmup must end at ctx_len
    if self.ctx_warmup and self.ctx_warmup[-1] != self.ctx_len:
        self.ctx_warmup = self.ctx_warmup + [self.ctx_len]
```

The validator catches all the "obvious typos" that would otherwise crash mid-training with cryptic errors.

## `effective_max_steps`

The `max_steps: "auto"` field is a string in the dataclass; it's resolved to an int at the first access:

```python
@property
def effective_max_steps(self) -> int:
    if self.max_steps == "auto":
        return self._chinchilla_solve()
    return int(self.max_steps)
```

`_chinchilla_solve()` computes the Chinchilla-optimal step count from the non-embedding param count and the per-step tokens. See [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) for the math.

## v5.8 opt-in research flags

Four new flags landed in v5.8. All default OFF — **profile before flipping**. The 5-run shpak comparison script is `uv run python tests/shpak_profile_5runs.py`.

| Flag | Section | Default | Effect on shpak 52.8M (batch=16 ctx=4096, 10 steps) |
|---|---|---|---|
| `sparse_6_8` | `model:` | `False` | +1 % step, +2 % mem. **No CUDA speed win** (no N:M-aware GEMM kernels); paper's main claim is **quality preservation** (1.58-bit is more sparsity-friendly than full-precision). Win on CPU/inference. |
| `selective_backward` | `model:` | `False` | When ON, activates LCSB (see `backward_ratio`). |
| `backward_ratio` | `model:` | `1.0` | Used when `selective_backward=True`. Practical: 0.3-0.7. **−44 % step, −25 % mem, +80 % tok/s at 0.5.** |

To flip LCSB (the recommended default after extended validation):

```yaml
# configs/default.yaml — shpak profile
model:
  selective_backward: true
  backward_ratio: 0.5
```

Don't combine `sparse_6_8: true` with `backward_ratio: 0.5` expecting a multiplicative speedup — shpak shows Sparse mask-computation overhead partially cancels LCSB's win. **🆕 v5.8**

### Pair-interaction overhead on top of LCSB alone (shpak 52.8M, 10 steps)

| Pair | Step overhead | Memory overhead |
|---|---:|---:|
| + Sparse-BitNet 6:8 | +6.4 % | +273 MB |

**LCSB alone is the recommended config** (1666 ms / 4102 MB / 39,322 tok/s on shpak). Don't add Sparse to LCSB without a specific reason (+6% step, +273 MB). Validate with `tests/v58_profile.py --mode shpak-pairs`. **🆕 v5.8, default ON in v6.0**

## How to add a new profile

1. Edit `configs/default.yaml`:

```yaml
my_profile:
  d_model: 640
  n_heads: 10
  n_layers: 20
  # ... override only the fields you want to change
```

2. Use it:

```bash
uv run train.py --profile my_profile
```

That's it. The base dataclass defaults are inherited, and the YAML overrides only the fields you specify.

## How to override on the CLI

Every field is a CLI flag. Dashes replace underscores:

```bash
uv run train.py --profile shpak \
                --d-model 768 \
                --n-layers 30 \
                --ctx-len 2048 \
                --max-steps 20000 \
                --lr 0.001 \
                --compile-mode reduce-overhead
```

Run `uv run train.py --help` for the full list.

## Common patterns

### Override per-GPU

```bash
# 8GB GPU: smaller batch, smaller ctx
uv run train.py --profile shpak --micro-batch-size 4 --ctx-len 2048

# 24GB GPU: larger batch, larger ctx
uv run train.py --profile shpak --micro-batch-size 32 --ctx-len 8192
```

### Continue a run with a smaller LR (anneal)

```bash
# Original run
uv run train.py --profile shpak --max-steps 20000 --lr 0.002
# Anneal: lower LR for last 20%
uv run train.py --profile shpak --resume checkpoints/ckpt_16000.pt --max-steps 20000 --lr 0.0002
```

### Reproduce a paper experiment

```bash
# Save the exact config
uv run train.py --profile shpak --save-config my_run.yaml
# Reproduce later
uv run train.py --config my_run.yaml
```

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselConfig` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The dataclass |
| `from_profile()` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | Profile loader |
| `from_yaml()` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | YAML loader |
| `effective_max_steps` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The `auto` resolver |
| `__post_init__` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The validator |
| `configs/default.yaml` | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | All 6 profiles |
| `test_config_validator` | [tests/test_config.py](file:///home/sehaxe/busel-ai/tests/test_config.py) | Compliance: d_model % n_heads == 0 |
| `test_chinchilla_solve` | [tests/test_config.py](file:///home/sehaxe/busel-ai/tests/test_config.py) | Compliance: 11M → 11_718 steps |

## See also

- [Profiles reference](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/profiles.md) — the user-facing guide
- [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) — Chinchilla solver
- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md) — the "first run" experience

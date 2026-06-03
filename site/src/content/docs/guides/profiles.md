---
title: Choosing a profile
description: Which of the bundled profiles (micro_test, quick_test, validation, chyzh, shpak, zubr) is right for you?
---

The `configs/default.yaml` file ships with six profiles. They trade off
**training time** vs **model capacity** vs **VRAM cost**. Pick one
based on what you want to do with the run.

## The six profiles at a glance

| Profile       | Total params | Active | Bit-size | Context | VRAM (bf16) | Time to Chinchilla (RTX 5060 Ti) | What it's for |
|---------------|-------------:|-------:|---------:|--------:|------------:|---------------------------------:|---------------|
| `micro_test`  | ~2 M         | ~1 M   | <1 MB    | 256 B   | ~0.5 GB     | seconds                          | smoke test / CI |
| `quick_test`  | ~3 M         | ~1.5 M | <1 MB    | 256 B   | ~0.6 GB     | minutes                          | quick sanity check |
| `validation`  | ~2 M         | ~1 M   | <1 MB    | 256 B   | ~0.5 GB     | ~1 min                           | pipeline smoke test |
| `chyzh`       | ~10 M        | ~5 M   | ~3 MB    | 512 B   | ~1.5 GB     | ~30 min                          | small-scale real training |
| `shpak`       | 52.8 M       | 25 M   | **11 MB**| 4096 B  | ~5 GB       | ~6 h                             | the "real" 50 M run |
| `zubr`        | 120 M        | 35 M   | 30 MB    | 16 384 B| ~12 GB      | ~24 h                            | long-context demo |

(All numbers are approximate — the actual cost depends on your data
loader, your batch size, and whether gradient checkpointing is on.)

## How to pick

### You just want to see *something* train

→ **`validation`**

```bash
uv run train.py --profile validation
```

200 steps, ~1 min on a 5060 Ti, loss drops 10.46 → 7.17, prints a
checkpoint. This is the smoke test for the whole pipeline: byte
loader, model, mAR, MTP, AutoPilot, optimizer, JSON log, checkpointing.

### You want a CI test

→ **`micro_test`** (faster than `quick_test`)

```bash
uv run train.py --profile micro_test --no-checkpointing
```

No real checkpoint produced. Designed to fail fast if any of the
imports or shape math is broken.

### You want a small model that actually does something

→ **`chyzh`** (~10 M, ~30 min)

```bash
uv run train.py --profile chyzh
```

Small enough to fit on a laptop, large enough to start producing
non-trivial perplexity on a small corpus. The Chinchilla auto-planner
gives it ~1 500 steps.

### You want the "main" run

→ **`shpak`** (52.8 M, ~6 h on a 5060 Ti)

```bash
uv run train.py --profile shpak
```

This is the profile the README advertises. Context grows 1024 → 2048
→ 4096 patches, batch adapts, ~25 000 steps, Chinchilla target ~3.84 B
byte-tokens. End result: an 11 MB checkpoint you can run inference on
with `tools/inference.py`.

### You want a long-context demo

→ **`zubr`** (120 M, context 16 384)

```bash
uv run train.py --profile zubr
```

The MLA latent cache makes 16 K contexts tractable on consumer
hardware. This profile is *expensive* — 24 hours on a 5060 Ti, ~12 GB
VRAM. Use it when you actually need long context.

## Tweak a profile without making a new one

All knobs are CLI flags on `train.py`:

```bash
uv run train.py --profile shpak \
    --no-compile            # disable torch.compile
    --no-checkpointing      # disable gradient checkpointing (more VRAM, faster)
    --compile-mode max-autotune   # try harder at compile time
    --resume checkpoints/shpak_step_10000.pt
```

Or edit `configs/default.yaml` directly — the format is plain YAML
and any field can be overridden. The auto-planner picks up your
`max_steps: "auto"` setting and computes the rest from the Chinchilla
byte-law `D ≈ 80 × N`.

## Making your own profile

Copy the closest existing profile in `configs/default.yaml` and tweak:

```yaml
my_profile:
  model:
    d_model: 256
    n_layers: 6
    n_heads: 4
    expert_hidden: 512
    num_experts: 4
    top_k: 2
    vocab_size: 259      # DO NOT change this — vocab=259 is byte-level
  data:
    data_path: "data_train"
    chunk_size: 1024
    batch_size: 64
  training:
    max_steps: "auto"    # computed from Chinchilla
    warmup_steps: "auto"
    min_lr_ratio: 0.1
    learning_rate_muon: 0.001
    learning_rate_adamw: 0.0001
    weight_decay: 0.1
    grad_accum_steps: 1
    checkpoint_interval: 500
```

Then run it with `uv run train.py --profile my_profile`.

:::caution
Do not change `vocab_size` from 259. The byte-level patcher is wired
to exactly 256 UTF-8 byte values + 3 multimodal specials (image marker,
PDF marker, padding). Any other value will silently misbehave.
:::

## Profile gotchas

- **`d_model % n_heads == 0`** — required for the multi-head layouts
  in `BusbaGDN2SeRoPEBlock` and `MultiHeadLatentAttention`.
- **`d_model % n_hyper == 0`** — required for the mAR `d_head` split
  (`d_head = d_model / n_hyper`).
- **`d_model` × `num_experts`** — budget VRAM. Each routed expert is
  a `d_model → expert_hidden → d_model` BitLinear pair, so memory is
  roughly `2 × d_model × expert_hidden × num_experts × 1.58 bits`.
- **`chunk_size % 4 == 0`** — the byte-to-patch stride is 4.
  The curriculum switches between `chunk_size ∈ {256, 1024, 4096}`.

The validator at the top of `buselConfig.__init__` (in `train.py`)
catches the first two of these and raises a `ValueError` with the
exact numbers, so you find out at startup, not at step 1 000.

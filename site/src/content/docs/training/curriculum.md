---
title: "Curriculum + Chinchilla planner"
description: "Sequence-length warmup (1024 → 2048 → 4096) and the Chinchilla D ≈ 80·N auto-planner that picks max_steps and warmup_steps for you."
sidebar:
  order: 4
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel's training has two curriculum-style mechanisms working together:

1. **Sequence-length warmup** — start at 1024 tokens, double every N steps until you hit the target. This is what lets the model "learn the byte" before being asked to use long-range context.
2. **Chinchilla auto-planner** — when `max_steps="auto"`, busel picks the step count and warmup length for you based on the [Chinchilla scaling law](https://arxiv.org/abs/2203.15556): `D ≈ 80 · N` (80 tokens per non-embedding parameter).

Both are built into `buselConfig` and wired up automatically in `train.py`. You don't have to think about them unless you want to override.

## Sequence-length warmup

```python
# training/curriculum.py
def next_ctx_len(step: int, current_len: int, target_len: int, doubler_every: int) -> int:
    if step > 0 and step % doubler_every == 0 and current_len < target_len:
        return min(current_len * 2, target_len)
    return current_len
```

The default doubling cadence is every 2000 steps:

| Step range | Context length | Profile |
|---|---|---|
| 0 → 2000 | 1024 | shpak, zubr, chyzh |
| 2000 → 4000 | 2048 | shpak, zubr, chyzh |
| 4000+ | 4096 | shpak, zubr, chyzh |

micro_test and quick_test skip the curriculum and stay at 512 / 1024.

### Why doubling, not linear?

Linear `ctx_len = base + slope·step` causes two problems:

1. The attention compute is O(S²) per layer, so linear ctx growth is non-linear in FLOPs.
2. At each step, the model is training on a *new* context length it has never seen. The gradient distribution shifts continuously, and the BitLinear STE doesn't get to "settle" on any one scale.

Doubling gives the model 2000 steps at each context length, which is enough to reach a local equilibrium. The transition cost is then a single O(S) attention re-compile, not a continuous re-tuning.

### Per-profile defaults

```yaml
# configs/default.yaml
shpak:
  ctx_len: 4096
  ctx_warmup: [1024, 2048, 4096]    # 2000 steps each

zubr:
  ctx_len: 4096
  ctx_warmup: [1024, 2048, 4096]    # 4000 steps each (slower doubling)

chyzh:
  ctx_len: 8192
  ctx_warmup: [1024, 2048, 4096, 8192]  # 8000 steps each
```

Override with `--ctx-warmup 1024,2048,4096,8192 --ctx-warmup-steps 4000` on the CLI.

## The Chinchilla auto-planner

When you write `--max-steps auto`, busel solves for the optimal step count:

```
D_optimal = 80 · N            (Chinchilla compute-optimal)
D = B · S · max_steps         (what the data loader actually delivers)
max_steps = (80 · N) / (B · S)
```

Where:
- `N` = non-embedding parameter count (counted by counting `p.numel() - 1` for each 2D param, summed)
- `B` = micro batch size
- `S` = effective sequence length (after curriculum, average over the run)
- `D` = tokens seen during training

### Worked example: Shpak on RTX 5060 Ti

```
N = 11.0M params, of which ~9.6M are non-embedding
B = 16, S = 4096 (target)
D_optimal = 80 · 9.6M = 768M tokens
max_steps = 768M / (16 · 4096) = 11_718 steps
warmup_steps = min(1000, max_steps // 10) = 1_000
```

The auto-planner then computes:
- `save_every = max_steps // 10` (10 checkpoints over the run)
- `eval_every = max_steps // 50` (50 evals)
- `cosine_end_fraction = 0.1` (final LR is 10% of base)

### When the auto-planner is wrong

Chinchilla's `D = 80·N` is the **compute-optimal** point. For longer runs (lower loss ceiling), you'd want `D = 200·N` or more. Override with:

```bash
uv run train.py --profile shpak --max-steps 25000   # ~170·N, fine for a quality push
```

For shorter runs (sanity check), `D = 20·N` works but expect higher loss. Below `D = 5·N` you're undertrained.

### What auto-planner does NOT consider

- **Data quality.** Chinchilla assumes uniform token informativeness; real corpora have ~3× variance. If your data is high-quality (curated books, code), the auto-planned steps are right. If it's web scrape, multiply by 1.5-2×.
- **1-bit specific overhead.** 1.58-bit converges ~30% slower per step than fp16 at the same N, because the ternary grid is less expressive. The auto-planner already accounts for this with the `0.3` factor built into `buselConfig._chinchilla_factor`.
- **Model architecture.** MoE gets 0.85× because routed experts need more training tokens to specialize. Dense models are 1.0×.

## CLI usage

```bash
# Auto-planned run
uv run train.py --profile shpak --max-steps auto

# Manual override
uv run train.py --profile shpak --max-steps 20000 --warmup-steps 500

# Disable Chinchilla, use the profile's default
uv run train.py --profile shpak
# (shpak profile has max_steps=12000 hard-coded in default.yaml)
```

The `buselConfig` validator runs at startup:

```python
# busel_config.py
@property
def effective_max_steps(self) -> int:
    if self.max_steps == "auto":
        return self._chinchilla_solve()
    return self.max_steps
```

## What the curriculum looks like in logs

`busel.log.jsonl` shows the curriculum transitions:

```json
{"ts": "...", "event": "curriculum", "step": 2000, "ctx_len": 1024, "next_ctx_len": 2048}
{"ts": "...", "event": "curriculum", "step": 4000, "ctx_len": 2048, "next_ctx_len": 4096}
{"ts": "...", "event": "curriculum", "step": 6000, "ctx_len": 4096, "next_ctx_len": 4096, "final": true}
```

You can grep for `"event": "curriculum"` to see the timing. The transition takes 1 step (recompile + the new dataloader batch).

## When to disable the curriculum

Three cases where you'd turn off ctx warmup:

1. **Continued pre-training** from a checkpoint already trained at `ctx_len=4096`. Set `--ctx-len 4096 --no-ctx-warmup`.
2. **Fine-tuning a long-context model** (Chinchilla, RoPE scaling). Set `--ctx-len 16384` directly.
3. **Benchmarking** where you want a fixed ctx length. Set `--ctx-len 2048` and disable the warmup schedule.

In all three, `train.py` will warn you that you're skipping the curriculum and ask for confirmation unless you pass `--no-ctx-warmup` (which makes it a hard skip, not a confirmation prompt).

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `next_ctx_len()` | [training/curriculum.py](file:///home/sehaxe/busel-ai/training/curriculum.py) | The doubling logic |
| `_chinchilla_solve()` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | `D=80N` solver |
| `effective_max_steps` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The `auto` resolver |
| `buselAutoPilot._curriculum_lr` | [training/autopilot.py](file:///home/sehaxe/busel-ai/training/autopilot.py) | Holds LR constant during ctx transitions |
| `test_chinchilla_solve_shpak` | [tests/test_curriculum.py](file:///home/sehaxe/busel-ai/tests/test_curriculum.py) | Compliance: 11M params → 11_718 steps |
| `test_ctx_warmup_doubling` | [tests/test_curriculum.py](file:///home/sehaxe/busel-ai/tests/test_curriculum.py) | Compliance: ctx_len doubles every N steps |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — how the curriculum fits in the main loop
- [AutoPilot v6.0](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md) — holds LR constant during ctx transitions
- [Profiles reference](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/profiles.md) — per-profile ctx defaults
- [Chinchilla paper](https://arxiv.org/abs/2203.15556) — the original scaling law

# PROJECT KNOWLEDGE BASE — busel (Бусел)

**Last updated:** 2026-06-05
**Branch:** main
**Test count:** 168 (unittest, no pytest)

## [PRIORITY] — read first
1. **Performance + LOC** — when in doubt, the faster + shorter option wins.
2. **Stage of project: EARLY.** Compatibility is NOT a constraint. Breaking changes are fine.
3. **Ship working code > elegant code.** We iterate on what runs, refactor later.
4. **Code is small** — entire model + training + data is ~3,000 LoC Python + ~140 LoC Rust. Do not add files without justification.

## OVERVIEW
**busel v5.8+** — Sovereign 1-bit (1.58b) Any-to-Text LLM. Hybrid Python + Rust (PyO3 via maturin). Targets consumer HW (RTX 5060 Ti 16 GB / Apple Silicon). Trained via CLI, documented in `site/` (Astro+Starlight, Bun).

**Architecture:** 1.58-bit ternary weights · byte-level vocab=326 · `stride=4` patching · 3:1 GDN-2:MLA attention · mAR residuals (Sinkhorn-Knopp on Birkhoff polytope) · **Top-1 MoE** with Blackboard Memory · MTP-4 heads · **LOTUS+Muon** (rank-8 factorised) + AdamW hybrid · **EMA of weights** · **selective activation checkpointing** (every=2) · **LCSB selective per-layer backward** (default ON in shpak/zubr/chyzh, v6.0) · **decoupled per-layer LR** (6 sub-groups) · **multi-stage pipeline** (pretrain → SFT → DPO → eval, v5.5) · **REPL tool executor** (v5.7) · **compile-safe checkpoint loader** (v5.7.1) · **opt-in v5.8 research feature**: Sparse-BitNet 6:8 (Dual STE, paper §3.3 quant-then-mask order).

## STRUCTURE
```
busel-ai/
├── model/             # BitNet v2 architecture (layers/attention/routing/backbone/patching/checkpoint)
├── training/          # LOTUS+Muon+AdamW hybrid, EMA, AutoPilot v6.0, MTP-4 loss, **stages/** (v5.5+)
├── data/              # Stream-interleaving token loader (list[int], Rust mmap or Python fallback)
├── multimodal/        # Any-to-token encoders (image/video/audio/PDF/docx) + 70-token special vocab
├── ui/                # Teto Vocaloid emoticon + rich terminal helpers
├── tools/             # Typer CLI (orchestrator, data_manager, plotter, inference, **tool_executor** v5.7)
├── tests/             # unittest suite (168) + ultra-stable profiler v2.1 + consolidated 3-mode v58_profile.py (v5.8)
├── busel_rust_io/     # PyO3 Rust ext: mmap ByteStreamer, ternary matmul, binary packer
├── configs/           # default.yaml — Shpak/Zubr/Chyzh/MicroTest/QuickTest/Validation profiles
├── site/              # Astro+Starlight docs (GitHub Pages)
├── checkpoints/       # *.pt + busel.log.jsonl (gitignored)
├── data_train/        # Raw training data (gitignored)
├── busel_registry.py  # 🛸 Plug-in extension-point registry (attention/optimizer/encoder/autopilot/curriculum/loss/stage)
├── busel_logging.py   # 📚 Structured JSONL event stream
├── train.py           # Legacy training orchestrator (v5.5+ also has cli.py pipeline)
├── cli.py             # Typer entrypoint (root-level — all user commands)
└── pyproject.toml     # uv-managed, maturin build backend
```

## DEFAULTS (buselPretrainConfig — single source of truth)
All flipped on by default. No opt-out. The whole arch is better for it.

| Field | Default | Was | Why |
|---|---|---|---|
| `optimizer_type` | `"lotus_muon"` | `"muon"` | Rank-8 LOTUS factorises Muon momentum — **~85× less optimizer state** |
| `top_k` (MoE) | `1` | `2` | 1 of N experts per token. **−35 % routed FFN FLOPs**, no quality loss |
| `use_ema` | `True` | `False` | EMA shadow of weights. **10-15 % fewer steps to same loss** |
| `ema_decay` | `0.999` | — | Standard EMA decay |
| `lotus_rank` | `8` | — | LOTUS rank-r. 6 = 60 % memory, 8 = 85 %, 16 = 95 % quality |
| `lotus_lr_scale` | `0.5` | — | LOTUS effective LR = `lr × lotus_lr_scale` |
| `lr_multipliers` | `{attn:1.0, ffn:1.0, mtp:1.0, norm:1.0, embed:0.5, router:0.5}` | `None` (single LR) | Decoupled per-layer LR — embed/router are noise-sensitive in 1-bit |
| `grad_ckpt_every` | `2` | `0` (off) | Selective activation checkpointing — **halves activation memory** at <5 % step-time cost |
| `selective_backward` (LCSB, 🆕 v6.0) | `True` (shpak/zubr/chyzh) | `False` | 50% of layers run under `no_grad` per forward; mAR identity still carries grad. **−44% step, −25% mem, +80% tok/s, 0 quality cost** on shpak. Off in test/calibration profiles (validation, micro_test, quick_test) for deterministic forward. |
| `backward_ratio` (LCSB, 🆕 v6.0) | `0.5` (LCSB on) | `1.0` | Used with `selective_backward=True`. Range: 0.3-0.7. |

## v6.0 OPT-IN RESEARCH FEATURES (defaults OFF — profile before flipping)
| Field | Default | Why OFF by default | Measured on shpak 52.8M |
|---|---|---|---|
| `sparse_6_8` (model) | `False` | Sparse-BitNet 6:8 (Dual STE, paper §3.3 quant-then-mask order, fixed in v6.0) — 2/8 weight sparsity. No N:M-aware CUDA kernels, so no speedup on training. CPU/inference wins. | +1% step, +2% mem (no win on CUDA) |

**Removed in v6.0:**
- **GradLite error feedback** (`use_error_feedback`) — LOTUS+bf16 round-trip is numerically exact → no error to feedback → framework is a no-op. **+1 GB VRAM overhead for 0% benefit.** All code, tests, and config lines deleted.

**Flipped to default ON in v6.0:**
- **LCSB selective per-layer backward** (`selective_backward=True, backward_ratio=0.5`) — validated at all 3 sizes (−57.7% step at 2M, −44.4% at 52.8M, −39.1% at 120M; 0 quality regression at 10 steps). Don't disable without measuring.

All 6 profiles in `configs/default.yaml` (validation, micro_test, quick_test, chyzh, shpak, zubr) inherit these defaults. The CLI `tests/profiler_run.py` defaults are aligned.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model layer | [model/](file:///home/sehaxe/busel-ai/model/AGENTS.md) | Must use `BitLinear_a4_8` or `H_BitLinear` (no raw `nn.Linear`) |
| Modify training loop | [training/](file:///home/sehaxe/busel-ai/training/AGENTS.md) | Pretrain is `buselPretrainStage`; SFT/DPO/eval in `training/stages/` |
| Add CLI command | [tools/](file:///home/sehaxe/busel-ai/tools/AGENTS.md) | Typer `@app.command`; subprocess pattern, never `import train` |
| Modify data loader | [data/AGENTS.md](file:///home/sehaxe/busel-ai/data/AGENTS.md) | Prefers Rust `ByteStreamer`; Python fallback exists |
| Add a new modality | [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) | `@register("encoder", ...)`; return `list[int]` in `[0, 326)` |
| Tune model size | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | 6 profiles: shpak / zubr / chyzh / micro_test / quick_test / validation |
| Profile step perf | [tests/AGENTS.md](file:///home/sehaxe/busel-ai/tests/AGENTS.md) | No `torch.profiler` (MPS hangs); use `tests/profiler_run.py` |
| Edit docs site | [site/AGENTS.md](file:///home/sehaxe/busel-ai/site/AGENTS.md) | Bun + Starlight; 7 stable URL sections |
| Add attention/optimizer/etc. | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | `@register("kind", "name")` — auto-discovered, no switch stmt |
| Customise terminal UX | [ui/](file:///home/sehaxe/busel-ai/ui/) | Teto emoticon + rich panels/spinners/trees |
| Consume event stream | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | `checkpoints/busel.log.jsonl` — one JSON per event |
| Add a new pipeline stage | [training/AGENTS.md](file:///home/sehaxe/busel-ai/training/AGENTS.md) | `@register_stage("name")` in `training/stages/<name>.py` |
| Load a checkpoint | [model/checkpoint.py](file:///home/sehaxe/busel-ai/model/checkpoint.py) | `load_state_dict_safely(model, sd)` — never raw `load_state_dict` |

## DO
- **Use the registry** for any plug-in point (attention, optimizer, encoder, autopilot, curriculum, loss, stage). `@register("kind", "name")` — no central switch statements.
- **Run 168 tests before pushing:** `uv run python -m unittest tests.test_suite` — all must pass.
- **Update README.md AND site/ docs** when a feature changes. The two-track rule: code change → README change → site/ change. Site/ is the human-friendly tour, README is the elevator pitch.
- **Match existing patterns.** Sample 2-3 similar files before adding a new one. Busel is small — patterns are visible at a glance.
- **Profile with `tests/profiler_run.py`** before claiming a speedup. Numbers, not vibes.
- **Use `uv run python cli.py`** — never `python cli.py` (maturin ext needs venv).
- **Add new files sparingly.** Default bias: extend an existing module. A 3,000 LoC codebase doesn't need more files.
- **Document anti-patterns in per-module AGENTS.md** when you discover one. The DO/NEVER list is the institutional memory.
- **Subprocess for train.py, direct call for everything else.** `tools/orchestrator.py` shells out; `model/`, `training/`, `data/`, `multimodal/` are in-process.
- **Single source of truth for checkpoint loading:** `model.checkpoint.load_state_dict_safely`. Four cross-config cases (compiled↔eager, save↔load) — the helper handles all of them.
- **Keep tests in `tests/test_suite.py`.** Never spawn a second test file. The suite is one big `unittest.TestCase` class.

## NEVER
- **BPE / tokenizers** — model is byte-level, vocab=326 (256 raw bytes + 70 plug-in specials). See [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) for the 70-token breakdown.
- **Raw `nn.Linear` in model** — must use `BitLinear_a4_8` or `H_BitLinear`. Breaks the 1.58-bit guarantee.
- **`H_BitLinear` for non-`o_proj`** — BitNet v2 spec mandates it for output projection only (massive-activation mitigation).
- **Disable any default in [DEFAULTS](#defaults-buselpretrainconfig--single-source-of-truth) without measuring** — LOTUS+Muon, EMA, Top-1, decoupled LR, selective ckpt, LCSB all help. Re-enabling them is the right move, not disabling.
- **`torch.profiler` on macOS** — known to hang; use `tests/profiler_run.py` (`time.perf_counter` + `cuda.max_memory_allocated`).
- **`model.load_state_dict(sd)` directly** — always `load_state_dict_safely(model, sd)`. Direct loads fail when saved with `--compile` (the default). The helper strips `_orig_mod.` and unwraps `OptimizedModule`.
- **Drop the `File` handle in `ByteStreamer`** — macOS mmap segfaults without it. Comment in `busel_rust_io/lib.rs:13` explains.
- **`cargo build`** — only `uv run maturin develop --release` produces a working Python ext.
- **PyO3 `unsafe` outside `Mmap::map`** — keep memory safety intact.
- **`import train` from `orchestrator.py`** — always `subprocess.run([sys.executable, "train.py", ...])`. The legacy `train.py` is reachable; the new pipeline is via `cli.py pipeline`.
- **`python-dotenv`** — project has its own `load_env()` parser in `tools/orchestrator.py`.
- **`assertTrue(x == y)` in tests** — use `assertEqual`. Better failure messages.
- **`nn.Embedding` for tokens** — use `nn.Parameter(torch.randn(vocab_size(), d_byte))`. Size auto-tracks the special-token registry.
- **Hardcoded vocab IDs** — import from `multimodal.special_tokens`. IDs are auto-allocated and may shift when tokens are added/disabled.
- **Hardcoded `259` in embedding shape** — use `vocab_size()`. (Old constant from before the 70-token expansion.)
- **Softmax on mAR logits** — Sinkhorn-Knopp projects to the Birkhoff polytope (doubly-stochastic), not softmax.
- **Muon on 1D params** (norms, biases) — `buselOptimizerEngine` filters them.
- **Muon on anything with `router` or `embed` in name** — `_MUON_EXCLUDE = ("router", "embed")`. Routers are policy, not value; embed is categorical.
- **Add `@torch.compile` to the whole `step()`** — only to inner Newton-Schulz function. Outer compile breaks the mAR FIFO buffer aliasing.
- **Change `momentum=0.95` of Muon** — spec is brittle; verify on validation profile if you must.
- **`F.cross_entropy` on CUDA when Liger is available** — 2-3× slower. Liger auto-falls back to vanilla.
- **`max_steps < warmup_steps` in any preset** — produces NaN spikes.
- **Skip the `Profile` step in `autopilot`** — HW profiling catches VRAM/RNG bugs early.
- **Execute a model-emitted tool call without user confirmation** — `tools/tool_executor.py:interactive_confirm` is the gate. `/auto on` is a per-session opt-in only.
- **Shrink `config.vocab_size` below `multimodal.special_tokens.vocab_size()`** — `buselModel.__init__` raises `ValueError`. Catches stale YAML.
- **Commit `data_train/`, `checkpoints/`, `.env`, `Cargo.lock`, `target/`** — all gitignored. The `.gitignore` is the single source of truth for what to never commit; do not duplicate that list in a NEVER rule.
- **Commit without explicit user request** — always wait for `commit` / `push` / `merge` instruction.
- **Enable `sparse_6_8` on layers with weight `numel() % 8 != 0`** — the `% 8` guard silently no-ops. Check shape before flipping.
- **Use `sparse_6_8=True` AND `backward_ratio=0.5` together expecting multiplicative speedup** — shpak 52.8M shows Sparse mask overhead (+6% step) partially cancels LCSB's win. Use LCSB alone. **🆕 v5.8, updated v6.0**

## UNIQUE STYLES
- **Emoji-prefixed module headers:** every Python file starts with `"""🦩 / ⚙️ / 💡 / 📚 / 🤖 / 🎯 / 🛸 ..."""` docstring.
- **Russian-language comments:** heavy Cyrillic throughout (technical).
- **`busel*` prefix** on all custom classes (`buselModel`, `buselOptimizerEngine`, `buselLossEngine`, `buselAutoPilot`, `buselPretrainStage`, …).
- **`cfg.profile` in checkpoint dict** — every saved `.pt` carries its profile name for auto-detect.
- **Rust parallel iterators** (`rayon::prelude::*`) for `ternary_matmul_cpu` (no GPU on inference).
- **Subprocess CLI orchestration** in `tools/orchestrator.py` — shells out to `train.py` and `tests/profiler_run.py` via `subprocess.run`.

## COMMANDS
```bash
# Setup
uv sync
uv add docling              # PDF support for data loader
uv run maturin develop --release   # Build Rust ext into venv

# Data
uv run python cli.py download-all --preset shpak
# (or copy PDFs/JSONL into data_train/ — auto-detected)

# Train
uv run python cli.py autopilot --profile shpak   # one-click: data + profiler + train
uv run train.py --profile shpak                   # manual (legacy)
uv run python cli.py pipeline --name pretrain-only # new (v5.5+)
uv run python cli.py profile                      # hardware profiler only

# Docs
cd site && bun install && bun run build           # GitHub Pages deploy

# Profile v5.8 research features (3 modes on shpak 52.8M + 3-size scaling)
uv run python tests/v58_profile.py --mode shpak-5run   # 4 configs on shpak (baseline / +Sparse / +LCSB / +Sparse+LCSB)
uv run python tests/v58_profile.py --mode shpak-pairs  # pair interactions (baseline / +LCSB / +Sparse+LCSB)
uv run python tests/v58_profile.py --mode scale-3sizes # 4 configs × 3 sizes (micro_test / shpak / zubr)
```

## PER-MODULE RULES
This file is the project-level summary. Module-specific rules, anti-patterns, and the per-class API live in per-module AGENTS.md:

- [model/AGENTS.md](file:///home/sehaxe/busel-ai/model/AGENTS.md) — BitLinear, H_BitLinear, GDN-2, MLA, mAR, MoE, MTP, compile-safe checkpoint loader
- [training/AGENTS.md](file:///home/sehaxe/busel-ai/training/AGENTS.md) — LOTUS+Muon routing, AutoPilot v6.0, loss engine, **stages/ framework** (v5.5)
- [data/AGENTS.md](file:///home/sehaxe/busel-ai/data/AGENTS.md) — Rust mmap streamer, Python fallback, multimodal dispatch
- [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) — 70-token vocab, 6 encoders (image/video/audio/PDF/docx/text)
- [tools/AGENTS.md](file:///home/sehaxe/busel-ai/tools/AGENTS.md) — Typer CLI, pipeline orchestrator, **REPL tool executor** (v5.7)
- [tests/AGENTS.md](file:///home/sehaxe/busel-ai/tests/AGENTS.md) — 168-test unittest suite, custom profiler, **consolidated 3-mode v58_profile.py** (v5.8)
- [busel_rust_io/AGENTS.md](file:///home/sehaxe/busel-ai/busel_rust_io/AGENTS.md) — PyO3 extension, mmap safety, Rayon threading
- [site/AGENTS.md](file:///home/sehaxe/busel-ai/site/AGENTS.md) — Astro+Starlight, build commands, URL structure

## NOTES
- **Checkpoint size guard:** reject `<10MB` `.pt` (corrupt) in `tools/inference.py`.
- **Target bit size:** 11 MB (Shpak) / 30 MB (Zubr) — 1.58-bit weights compress ~10× vs fp16.
- **Metrics log:** `checkpoints/metrics.jsonl` (one JSON per step, for ETA).
- **Event stream:** `checkpoints/busel.log.jsonl` — structured JSON for downstream (TG bot, web). Events: `training_start`, `model_initialized`, `chinchilla_planned`, `curriculum_upgrade`, `step_complete`, `checkpoint_saved`/`checkpoint_rejected`/`checkpoint_failed`, `emergency_save_requested`, `emergency_checkpoint`, `stage_complete`, `pipeline_start`/`pipeline_complete`, `stage_failed`, `training_complete`, `autopilot`.
- **Registry kinds:** `attention` (`gdn2`, `mla`), `optimizer` (`muon`, `lotus_muon`, `hybrid_muon_adamw`), `autopilot` (`v6`), `curriculum` (`doubling`), `encoder` (`image`, `video`, `audio`, `pdf`, `docx`, `text`), `loss`, `stage` (`pretrain`, `sft`, `dpo`, `eval`).
- **Teto emoticon cycle:** 12-frame kawaii idle loop (`(ᗜˬᗜ)`, `ξ(｡•̀ᴗ-)✧ξ`, `ξ(≧◡≦)ξ`, `▼ᗜˬᗜ▼`, …) — see `ui.teto.frames()`. States: `idle`, `blink`, `smile`, `think`, `wave`, `training`, `done`.
- **macOS Rust flag:** `.cargo/config.toml` uses `link-arg=-undefined,dynamic_lookup` for macOS.
- **License:** CC BY-NC-SA 4.0 (NC clause — NO commercial use). Contact `sehaxe` for commercial licence.
- **LOTUS paper:** arXiv:2602.01233. Rank-r factorised Muon momentum.
- **Muon repo:** github.com/KellerJordan/Muon — original implementation.

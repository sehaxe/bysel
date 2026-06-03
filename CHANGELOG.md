# Changelog

All notable changes to Busel are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

---

## [5.2.0] — 2026-06-04 — "Sovereign 1-bit LLM"

### Added

- **1.58-bit BitLinear + H_BitLinear** for the entire backbone
  (1 ternary weight, INT4/INT8 activations, H_BitLinear for `o_proj` only).
- **mAR (Manifold Constrained Attention Residuals)** — `n_hyper` parallel
  residual streams, multi-query attention between current activation and
  each stream, projected onto the **Birkhoff polytope** via
  `n_sinkhorn_iters` of Sinkhorn-Knopp. Identity-initialised
  (`+5.0` diagonal bias) so it starts as a no-op.
- **3:1 GDN-2 / MLA attention mix.** GDN-2 uses Triton `fla.ops.gdn2` when
  available, with a JIT-fallback. MLA compresses KV to `d_c=128`.
- **MoE with Blackboard Memory** — 2 always-on shared experts + N routed
  (Top-2), with gate/read BitLinear enrichments before the router.
- **MoD router** with `capacity_factor` (currently 1.0 = full sequence).
- **Multi-Token Prediction (MTP-4)** — 4 parallel heads, decaying loss
  weights `[1.0, 0.5, 0.25, 0.125]`. Heads share the MTP embed weight.
- **Hybrid Muon + AdamW optimizer** — 2D `proj` params without `router`
  in the name → Muon (Newton-Schulz ×5, scale `0.2·√max(A,B)`); the rest
  → AdamW. Auto-falls-back to **FlashMuon** (Triton) when available.
- **buselAutoPilot v6.0** — predictive 3σ dampening, adaptive gradient
  clipping, dynamic weight-decay curve.
- **Curriculum learning** — context 64 → 128 → 256 patches, batch adapts
  inversely to keep VRAM constant.
- **Chinchilla auto-planner** — `D ≈ 80 × N` byte-tokens, divided by
  `batch × ctx/4` to derive `max_steps` and `warmup_steps`.
- **Gated FastBLT patcher** — byte-level conv with stride=4, sigmoid-gated.
  `vocab_size=259` (256 bytes + 3 multimodal specials).
- **Rust mmap byte streamer** (`busel_rust_io/`) — zero-copy large-file
  reads with `rayon` parallel iterators for the ternary CPU matmul path.
- **Multimodal encoding** — `byte=256` marker for inline images, PDF
  parsing via Docling (optional), JSONL + Parquet support.
- **CLI surface** (`cli.py` via `tools/orchestrator.py`) —
  `download-all`, `autopilot`, `profile`, `inference`, `repl`, `plot`.
- **Teto UI module** (`ui/`) — Kasane Teto 12-frame emoticon cycle + rich
  terminal helpers (gradient text, animated header, spinner, progress bar,
  stats table, project tree). Auto-falls-back to plain `print` without rich.
- **Plug-in registry** (`busel_registry.py`) — `@register("kind", "name")`
  decorator with thread-safe collision detection and an `override=True`
  opt-in. Currently registered:
  - `attention/gdn2`
  - `attention/mla`
  - `optimizer/muon`
  - `optimizer/hybrid_muon_adamw`
- **Structured JSONL event log** (`busel_logging.py`) — append-only
  stream of all training events to `checkpoints/busel.log.jsonl`.
  Idempotent on resume; schema documented in the README.
- **Starlight docs site** (`site/`) — Astro + Starlight, deployed to
  GitHub Pages. Has a sidebar with Architecture / Training / Data / API /
  Performance / Operations sections.
- **61 unit tests** in `tests/test_suite.py` covering: paper compliance
  (BitNet, mAR, mHC, AttnRes, GDN-2, MLA, MoE, MTP, Muon), end-to-end
  integration, registry, logging, and the Teto UI helpers.

### Changed

- **README is now in English** and serves as the navigation hub
  (links to docs site, AGENTS.md, the registry, the event log).
- **`train.py` --compile-mode flag** — `default | reduce-overhead |
  max-autotune` (was implicit `default` only). Robust error-handling
  with auto-fallback to default if the requested mode fails.
- **`train.py` SIGINT handler is now a flag-setter**, not an immediate
  `state_dict() + torch.save()`. The save runs at the next safe step
  boundary to avoid the `FakeTensor` crash when SIGINT fires during
  `torch.compile` tracing (initial compile or any shape-triggered
  recompile). Fixes the `AssertionError: Please convert all Tensors to
  FakeTensors first` crash that could happen on Ctrl-C.
- **Per-AGENTS.md** — `model/`, `training/`, `data/`, `tests/`, `tools/`,
  `busel_rust_io/` each have a knowledge-base file covering scope,
  where-to-look, key classes, conventions, anti-patterns, notes.
- **Top-level AGENTS.md** — single source of truth for project layout,
  command cheatsheet, license.

### Fixed

- `torch.compile` + SIGINT during compile/recompile → `FakeTensor`
  `AssertionError` on `param.detach()`. Now deferred to safe step.
- `_orig_mod.` prefix in state dict after `torch.compile` — stripped on
  resume (`_strip_compile_prefix`) and on the deferred emergency save.
- Compile mode is now configurable; non-default modes that fail
  (e.g. `reduce-overhead` on this architecture due to mAR stream
  aliasing) fall back to `default` automatically.

### Removed

- `telegram_bot/` and all `aiogram` references (planned as a separate
  future repo).
- `services/` (FastAPI serve) and all `fastapi` / `uvicorn` /
  `INFERENCE_API_URL` references (planned as a separate future repo).
- `docs/` (legacy Starlight site) — replaced by the new `site/` with
  the comprehensive wiki.

### Anti-patterns (do not violate)

- **NEVER** use BPE / tokenizers. Vocab is exactly 259.
- **NEVER** add raw `nn.Linear` outside `BitLinear_a4_8`.
- **NEVER** checkpoint `*.pt < 10 MB` — auto-rejected as corrupt.
- **NEVER** use `torch.profiler` on macOS — use `tests/profiler_run.py`.
- **NEVER** mix `H_BitLinear` for non-`o_proj` outputs.
- **NEVER** bypass `BitLinear_a4_8` `is_intermediate=True` in FFN experts.
- **NEVER** commit `data_train/`, `checkpoints/`, `.env`, `Cargo.lock`,
  `uv.lock`.

---

## [5.1.0] — earlier internal

Internal milestone that added the BitLinear + mAR + GDN-2 foundation.
Pre-dates this changelog.

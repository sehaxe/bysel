# Changelog

All notable changes to Busel are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

---

## [5.5.0] — 2026-06-04 — "Stage Framework Foundation" 🛸

### Added

- **`training/stages/` module** — plug-in multi-stage training pipeline
  framework. Stages are registered via `@register_stage("name")` (a new
  `busel_registry` kind) and implement the `BaseStage` Protocol
  (`setup(cfg) → run(state) → finalize(state)`).
  - `training/stages/__init__.py` — public API exports; eagerly imports
    `pretrain` to trigger registration on package import
  - `training/stages/base.py` — `BaseStage` Protocol, `StageState`/
    `StageSpec`/`PipelineConfig` dataclasses, `register_stage` decorator,
    `load_pipeline_yaml()` validator (rejects unknown stage names,
    missing keys, malformed shapes)
  - `training/stages/pretrain.py` — `buselPretrainStage` extracted from
    `train.py:main()`. Behavior is preserved 1:1 (chinchilla planner,
    curriculum warmup, autopilot, gradient checkpointing, torch.compile,
    MTP-4 targets, scheduled checkpoints, final checkpoint, SIGINT
    emergency save). `buselPretrainConfig.from_profile()` parses a
    YAML profile dict.
- **`configs/pipelines/pretrain-only.yaml`** — minimal 1-stage pipeline
  preset equivalent to `uv run train.py --profile shpak`. Shows the
  pipeline YAML schema (`name` + `stages[]` + optional `global_params`).
- **`cli.py pipeline` subcommand** — new Typer entrypoint that runs a
  multi-stage pipeline. Loads YAML from `configs/pipelines/<name>.yaml`,
  instantiates stages via `get_stage(name)`, calls `setup → run →
  finalize` in order. Supports `--start-stage` to resume mid-pipeline
  and `--config-dir` to override the preset path. Logs `pipeline_start`/
  `stage_start`/`stage_complete`/`pipeline_complete` events to
  `checkpoints/busel.log.jsonl`.
- **`busel_registry.register("stage", name)`** — new registry kind
  alongside the existing `attention`/`optimizer`/`encoder` kinds.
  `get_stage("pretrain")`, `list_stages()`, `is_stage_registered(name)`
  are the public read API.
- **14 new tests** in `tests/test_suite.py` (prefix `STG-1` … `STG-14`):
  registration, retrieval, unknown-stage rejection, lifecycle methods,
  config parsing (valid + 2 invalid shapes), YAML loading
  (valid/missing/unknown-stage/4 missing-key shapes), dataclass fields,
  orchestrator command import. All guarded by `HAS_TRAINING_STAGES` so
  they degrade gracefully on broken imports.
- **2 existing tests fixed** — `test_registry_decorator_basic`,
  `test_registry_collision_raises`, `test_registry_override_allowed`
  no longer call `clear_registry()` (which was wiping production
  entries needed by later tests). They now use `unregister(test_kind,
  name)` for surgical cleanup of their own namespace.

### Changed

- **`tools/orchestrator.py`** — added `pipeline(name, start_stage,
  config_dir)` Typer command. Kept the legacy `train`/`autopilot`/
  `profile` commands unchanged.
- **`cli.py`** — registered `pipeline` subcommand; bumped module
  docstring from v4.1 to v5.5.
- **`training/AGENTS.md`** — added a complete section on `stages/`
  (STRUCTURE, WHERE TO LOOK, KEY CLASSES, CONVENTIONS, ANTI-PATTERNS,
  NOTES), including the eager-import-in-`__init__.py` pattern that
  triggers `@register_stage` registration on package import.
- **`tools/AGENTS.md`** — documented the new `pipeline` command,
  pipeline YAML schema, and the connection to `training/stages/`.

### Backward Compatibility

- **`train.py` is UNCHANGED.** Users can run `uv run train.py --profile
  shpak` (legacy) OR `uv run cli.py pipeline --name pretrain-only`
  (new); both produce equivalent checkpoints. The `train.py` → stage
  migration is a separate PR.

### Anti-patterns (do not violate — new for 5.5.0)

- **NEVER** import `train.py` from `stages/` — `buselPretrainStage` is
  the new canonical interface. The legacy `train.py` stays untouched.
- **NEVER** register a stage in a runtime-loaded module without
  re-triggering `__init__.py` — the registry is populated only at
  import time. New stages MUST add their module import to
  `training/stages/__init__.py` to be discoverable.
- **NEVER** swallow `KeyError` from `get_stage()` in production — the
  orchestrator treats it as a hard config error.
- **NEVER** use `clear_registry()` in tests that don't own the entire
  registry state — it wipes production entries (`gdn2`, `mla`, `muon`,
  `hybrid_muon_adamw`, `pretrain`) needed by later tests. Use
  `unregister(kind, name)` for surgical cleanup instead.

### Performance

No regression. `buselPretrainStage` is a refactor of `train.py:main()`,
not a new implementation — same training loop, same optimizer, same
checkpoint format. End-to-end benchmark on RTX 5060 Ti (shpak profile,
200 steps): same tok/s as 5.4.0.

### Roadmap (v5.6+)

- **v5.6** — SFT stage + HF dataset downloader
- **v5.7** — DPO stage + safety/honesty/critical-thinking data
- **v5.8** — Eval stage (perplexity, code, format, honesty probes)
- **v5.9** — REPL stage (chat template, streaming, tool loop)
- **v6.0** — Full pipeline preset (`configs/pipelines/full.yaml`)
  + `train.py` deprecation

---

## [5.4.0] — 2026-06-04 — "Sovereign 70-token Vocabulary" 🛸

### Added

- **`multimodal/special_tokens.py`** — a plug-in `SpecialToken` registry.
  Frozen dataclass with `name`, `id`, `layer`, `description`, `enabled`;
  int-coercible. Auto-allocates IDs starting at 259, exposes:
  - `vocab_size()` — currently **326** (256 bytes + 3 legacy + 67 plug-in)
  - `enabled_ids()` — 70 ints for the inference logits mask
  - `get_special_token(name)`, `register_special_token(name, layer, description)`
  - `disable_special_token(name)`, `enable_special_token(name)`
  - `list_special_tokens()`, `layer_summary()` — introspection
  - Self-test on `python -m multimodal.special_tokens` prints the full layer
    breakdown and validates the toggle invariant
- **70 special tokens across 12 functional layers** — auto-defined at import:
  1. **sequence** (4) — `BOS`, `EOS`, `PAD`, `UNK`
  2. **modality** (6) — `MOD_IMAGE`, `MOD_VIDEO`, `MOD_AUDIO`, `MOD_PDF`, `MOD_DOCX`, `MOD_TEXT`
  3. **mm_struct** (3) — `FRAME_SEP`, `AUDIO_CHUNK_SEP`, `CHANNEL_SEP`
  4. **role** (4) — `ROLE_SYSTEM`, `ROLE_USER`, `ROLE_ASSISTANT`, `ROLE_TOOL`
  5. **reasoning** (4) — `THINK_START`, `THINK_END`, `PLAN_START`, `PLAN_END`
  6. **code** (4) — `CODE_BLOCK_START`, `CODE_BLOCK_END`, `DIFF_START`, `DIFF_END`
  7. **tool_xml** (12) — Anthropic-style `<function_calls>` / `<invoke>` / `<parameter>` / `<result>` envelope (start/end per tag × 6)
  8. **tool** (12) — opencode tool vocabulary: `TOOL_BASH`, `TOOL_READ`, `TOOL_WRITE`, `TOOL_EDIT`, `TOOL_GREP`, `TOOL_GLOB`, `TOOL_FETCH`, `TOOL_SEARCH`, `TOOL_TASK`, `TOOL_TODO`, `TOOL_LSP`, `TOOL_ASK`
  9. **task** (4) — `TODO_START`, `TODO_END`, `TASK_DONE`, `TASK_PENDING`
  10. **reference** (6) — `FILE_PATH_START`/`END`, `URL_START`/`END`, `CITE_START`/`END`
  11. **subagent** (4) — `SUBAGENT_START`/`END`, `SUBAGENT_RESULT_START`/`END`
  12. **status** (4) — `STATUS_SUCCESS`, `STATUS_ERROR`, `STATUS_TIMEOUT`, `STATUS_CANCELLED`
- **`buselModel.__init__` vocab sanity check** — raises `ValueError` with
  a helpful diagnostic if `config.vocab_size < vocab_size()`. Catches
  stale yaml configs that forgot to bump the vocab.
- **13 new tests** in `tests/test_suite.py` (prefix `MM-14` … `MM-26`):
  registry correctness, layer-summary sanity, all 6 encoders emit
  `MOD_*` prefix, disable/enable roundtrip preserves IDs, runtime token
  registration grows vocab, legacy-collision rejection, decoder accepts
  both `MOD_*` and legacy markers, patcher `embed_weight` shape tracks
  registry, `buselModel` rejects undersized config.
- **Smoke test** `smoke_test_vocab326.py` — end-to-end forward + backward
  + optimizer loop on real GPU, verifies all 4 MTP heads are finite,
  gradients flow, logits cover full vocab=326, special tokens reach
  training targets.

### Changed

- **`configs/default.yaml`** — all 6 profiles: `vocab_size: 259` → `326`
  (with inline comment: `# 256 raw bytes + 3 legacy + 67 plug-in specials`).
- **`model/patching.py`** — `embed_weight` is now
  `nn.Parameter(torch.randn(vocab_size(), d_byte))` (was `(259, d_byte)`),
  auto-tracking the registry. Old hardcoded 259 references removed.
- **`model/backbone.py:buselModel`** — uses dynamic `vocab_size()` when
  `config.vocab_size` is 0, and sanity-checks against the registry.
- **`multimodal/encoders.py`** — `_resolve_modality_marker()` returns
  `int` (was `SpecialToken`). All 6 encoders now emit `MOD_*` prefixes
  for modality awareness. `TextEncoder.encode_file` now emits
  `[MOD_TEXT, *bytes, MEDIA_END]` (was bare bytes). Legacy `[256, ...,
  257]` is still accepted by the decoder for backward compat.
- **`tools/inference.py`** — `apply_sampling` masks
  `logits[256:vocab_size()] = -inf` (was `logits[256:]`).
- **`data/pipeline.py`** — JSONL image rows emit
  `[MOD_IMAGE, *payload, MEDIA_END]` (was `[256, ..., 257]`). Has a
  graceful fallback to legacy markers if `multimodal` is unavailable.
- **7 existing tests fixed** to use `vocab_size()` (dynamic) and
  modality-specific markers (`MOD_IMAGE`, `MOD_VIDEO`, etc.) instead
  of the old `256` / `IMAGE_MARKER` constants.
- **`multimodal/AGENTS.md`**, **`model/AGENTS.md`**, **`data/AGENTS.md`**
  — all updated to document the new 70-token vocabulary, the
  `MOD_*` modality markers, the dynamic vocab API, the new ANTI-PATTERNS
  (no hardcoded token IDs, no shrinking `config.vocab_size`), and the
  breaking checkpoint-incompatibility note.

### Breaking Changes

- **Old 259-vocab checkpoints are NOT loadable** in v5.4.0. The
  `embed_weight` shape is now `(326, d_byte)`. Loading a `(259, d_byte)`
  checkpoint will fail with strict-state-dict mismatch. **Re-train from
  scratch** or convert via the registry's `disable_special_token` +
  `register_special_token` API to match the old ID layout.

### Anti-patterns (do not violate — new for 5.4.0)

- **NEVER** hardcode token IDs (`256`, `MOD_IMAGE.id`, etc.) — always
  import from `multimodal.special_tokens`. The IDs are auto-allocated
  and may shift when tokens are disabled or added.
- **NEVER** set `config.vocab_size` smaller than `vocab_size()` —
  `buselModel.__init__` will reject it.
- **NEVER** shrink `config.vocab_size` to "remove" disabled tokens — the
  registry keeps the ID slot reserved; the inference mask only covers
  enabled IDs.
- **NEVER** use the old `256` / `IMAGE_MARKER` constants in new code —
  use `MOD_IMAGE` / `MOD_VIDEO` / `MOD_AUDIO` / `MOD_PDF` / `MOD_DOCX` /
  `MOD_TEXT`. Legacy 256-258 are kept for backward-compat decode only.

### Performance

No regression. Embedding went from `(259, 128)` to `(326, 128)` —
+67×128 = +8.5 K params for the validation profile (0.4 % of total).
Verified on RTX 5060 Ti: 5 forward + backward + step cycles, no NaN/Inf,
no measurable throughput change.

---

## [5.3.0] — 2026-06-04 — "Multimodal Sovereign" 🛰️

### Added

- **`multimodal/` module** — 6 encoders that turn any file into a
  `list[int]` stream in the same 259-vocab the byte-level model already
  consumes. All registered via `@register("encoder", name)`:
  - `ImageEncoder` — image (PNG/JPEG/WebP/BMP/GIF/TIFF) → 32×32×3 RGB payload
  - `VideoEncoder` — video (MP4/MOV/AVI/MKV/WebM) → max 8 evenly-spaced 32×32 frames
  - `AudioEncoder` — audio (WAV/FLAC/OGG) → 16-bit PCM with sr/n/sw header
  - `PDFEncoder` — PDF → Docling markdown → UTF-8 bytes
  - `DocxEncoder` — DOCX → python-docx plain text → UTF-8 bytes
  - `TextEncoder` — UTF-8 pass-through (no markers)
- **`download-multimodal` CLI** (`uv run python cli.py download-multimodal --limit 8`)
  generates a synthetic 4-modality test set in `data_train/multimodal/` —
  image / video / audio / docx — and writes a `multimodal_manifest.jsonl`.
  No internet required.
- **OpenCV (cv2) fast paths** — `ImageEncoder` and `VideoEncoder` use
  `cv2.imread` + `cv2.resize INTER_AREA` + `cv2.VideoCapture` (with
  `CAP_PROP_FRAME_COUNT` for O(1) frame count + `cap.grab()` for
  seek-skipping). PIL and imageio are fallbacks (~5-10× slower).
- **Multimodal docs page** at
  `site/src/content/docs/data/multimodal.md` — complete guide including
  token layouts, encoder dispatch, performance benchmarks, and the
  rationale for byte-level uniformity.
- **`multimodal/AGENTS.md`** — module-level knowledge base following
  the convention of `data/`, `model/`, `training/`, `tools/`,
  `busel_rust_io/`, `tests/`.
- **13 new tests** in `tests/test_suite.py` (prefix `MM-1` … `MM-13`):
  registry, image/video/audio/docx/text round-trips, marker validation,
  fixed-point losslessness, end-to-end pipeline collate, cv2 fast-path
  throughput benchmarks (100 images <500 ms, 60-frame video <2 s).
- **Multimodal training verified end-to-end on RTX 5060 Ti** — 8 steps
  on COCO images + captions, loss 5.59 → 5.49, no NaN, 97.5 % params
  to Muon (the routing fix from 5.2.0). Markers 256/257 correctly placed
  in every batch.

### Changed

- **`data/pipeline.py:buselOmnivoreTextExtractor`** auto-dispatches
  image/video/audio/PDF/docx files to the new encoders. The
  `self.raw_bytes` is now `list[int]` (was `bytearray`) — required
  to hold token values ≥ 256.
- **`multimodal/encoders.py` docstring** documents the critical
  design: encoders return `list[int]`, NOT `bytes`, because Python's
  `bytes` cannot represent values ≥ 256.
- **AGENTS.md / data/AGENTS.md** — updated to reflect the new
  representation, the cv2 fast path, and the resolved latent bug
  (`bytearray.append(256)` would have raised `ValueError`).

### Fixed

- **Latent bug in `buselOmnivoreTextExtractor`** — `bytearray.append(256)`
  raises `ValueError: byte must be in range(0, 256)`. The image path
  in JSONL was never exercised in tests, so the bug was hidden. Fixed
  by switching `self.raw_bytes` to `list[int]`. The collate function
  `collate_busel_batch` already supported `list` input.
- **Tokenizer marker documentation drift** — the previous `multimodal.md`
  page documented `__BOS__=256`, `__DOC_SEP__=257`, `__MEDIA__=258`,
  but the model and pipeline actually use `__MEDIA_START__=256`,
  `__MEDIA_END__=257`, `__DOC_SEP__=258`. The page has been rewritten
  to match the implementation.

### Performance (RTX 5060 Ti)

| Operation | Latency | vs PIL baseline |
|---|---:|---:|
| Image encode (256² → 32×32) | **0.44 ms** | 5.7× faster |
| Video encode (60 frames @ 128×128 → 8 frames) | **4.5 ms** | ~10× faster |
| Audio encode (1 s @ 16 kHz WAV) | ~0.5 ms | soundfile baseline |
| End-to-end multimodal pipeline (8 files, mixed) | 14.8 ms | — |

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

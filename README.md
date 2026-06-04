# Busel (Бусел) — Sovereign Any-Scale 1-bit Multimodal LLM

> *Pronounced **[ˈbusɛl]** — from Belarusian **бусел** (stork).*
>
> A token-free, 1.58-bit, hybrid linear-attention LLM with **mAR** residuals,
> MoE, byte-level patching, and MTP-4. **Any-to-text**: trains on images,
> video, audio, PDF, docx in the same byte stream as text. Runs on consumer
> hardware (RTX 5060 Ti 16 GB / Apple Silicon Mac) without any external
> tokenizer.

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)
[![CUDA / MPS / CPU](https://img.shields.io/badge/device-CUDA%20%7C%20MPS%20%7C%20CPU-green.svg)](#hardware-support)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg)](./LICENSE)
[![Docs: Starlight](https://img.shields.io/badge/docs-starlight-purple.svg)](https://sehaxe.github.io/busel-ai/)

---

## Why Busel exists

Modern LLMs are big, expensive, and bound to GPUs. Busel is an experiment
in whether the *scaling-laws ceiling* can be pushed down by:

1. **1.58-bit ternary weights** — every linear layer quantizes to `{-1, 0, +1}`
   in the forward pass (real weights are kept in a master copy with STE updates).
   At inference the math becomes pure additions on CPU, the model shrinks to
   ~11 MB for a 50 M-param profile, and FP16 multiplications disappear.
2. **Byte-level tokens (vocab=259)** — no BPE. No 40 % of the model wasted on
   an embedding matrix. The same token stream carries text, code, JSON,
   images (32×32×3 = 3072 tokens, token `256` marker), video, audio, PDF
   (via Docling), and docx. See [Multimodal data](https://sehaxe.github.io/busel-ai/data/multimodal/).
3. **mAR (Manifold Constrained Attention Residuals)** — replaces the
   classical `y = x + f(x)` residual with an input-dependent, doubly-stochastic
   mixture of the last `n_hyper` layer outputs projected onto the **Birkhoff
   polytope** (Sinkhorn-Knopp). This is the combination of
   [Kimi Attention Residuals](https://arxiv.org/abs/2603.15031) and
   [DeepSeek mHC](https://arxiv.org/abs/2512.24880) called
   **mAR** in Busel.
4. **3:1 GDN-2 / MLA attention** — 75 % of layers are linear (GDN-2,
   O(1) cache); 25 % are MLA (latent KV, `d_c=128`). A 128 K-token context
   uses ~98 MB of MLA cache.
5. **Hybrid Muon + AdamW** — `proj` weights go through the Muon Newton-Schulz
   orthogonaliser; everything else (norms, biases, embeddings, router) uses
   AdamW. Plus `buselAutoPilot` v6.0 — predictive 3σ dampening, adaptive
   gradient clipping, dynamic weight decay.
6. **Curriculum + Chinchilla auto-planner** — context grows 64 → 128 → 256
   patches, batch adapts inversely to hold VRAM constant, and the exact step
   count is derived from the Chinchilla byte-law
   `D ≈ 80 × N` for the profile in use.
7. **OpenCV fast paths** — image and video encoders use cv2 (`cv2.resize`
   with `INTER_AREA`, `cv2.VideoCapture` with `CAP_PROP_FRAME_COUNT` +
   `cap.grab()` for seek-skipping). ~5-10× faster than PIL/imageio.

The codebase is intentionally small — the entire model + training + data
pipeline is ~2,500 lines of Python and ~140 lines of Rust, so you can read
the whole thing in an afternoon.

---

## Quick start

```bash
# 1. Install Python deps (uses uv)
uv sync

# 2. (Optional) Add PDF + vision support
uv add docling              # PDF support
# opencv-python-headless is already in pyproject.toml (image/video fast path)

# 3. Compile the Rust byte-streamer into the venv
uv run maturin develop --release

# 4. (Optional) Drop your own data into data_train/, OR use the 1-click data + train workflow:
#    a) Download all 4 HF data presets (3 SFT + 1 DPO):
uv run python cli.py download-data
#    b) Run the full multi-stage pipeline (pretrain → SFT → DPO → eval):
uv run python cli.py train-all

# 5. Or train a single profile the legacy way:
uv run train.py --profile shpak

# 6. (Optional) Drop multimodal files (image/video/audio/docx) for any-to-text:
uv run python cli.py download-multimodal --limit 8
```

That's the whole pipeline. The first run is slow because `torch.compile`
traces the model (≈30 s on RTX 5060 Ti for a small profile, longer for
Shpak). Subsequent steps run at ~570 k tok/s on the validation profile.

For multimodal training, drop images / videos / audio / PDFs / docx files
into `data_train/multimodal/` (or any subfolder of `data_train/`). The
data loader auto-dispatches by extension.

---

## Architecture in one minute

```
                text / image / video / audio / PDF / docx
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │ multimodal/encoders                │  OpenCV fast path
            │   text  → list[int]  (UTF-8 bytes)  │  PIL/imageio fallback
            │   image → [256] [3072 RGB] [257]   │  docling for PDF
            │   video → [256] [N×3072 frames]    │  python-docx for docx
            │   audio → [256] [sr][n][sw][PCM]   │  soundfile for audio
            └─────────────────────────────────────┘
                              │  tokens (B × T, values 0-258)
                              ▼
            ┌─────────────────────────────────────┐
            │ StridedFastBLTPatcher              │  stride=4 conv
            │   vocab=259 → d_byte=128 → d_model │  + sigmoid gate
            └─────────────────────────────────────┘
                              │  patches (B × T/4 × d_model)
                              ▼
            ╔═════════════════════════════════════╗
            ║  buselModel (n_layers decoder)      ║
            ║                                     ║
            ║   for L in 1..n_layers:             ║
            ║     h = ManifoldConstrainedAttnRes  ║  ← mAR: Sinkhorn on
            ║         (current + n_hyper streams)║     Birkhoff polytope
            ║     h = buselDecoderLayer(h)        ║  ← attn (GDN-2 or MLA)
            ║                                     ║     + MoE (2 shared +
            ║                                     ║       N routed, Top-2)
            ╚═════════════════════════════════════╝
                              │  hidden (B × T/4 × d_model)
                              ▼
            ┌─────────────────────────────────────┐
            │ buselMTP4Pipeline (4 heads)         │  predict t+1..t+4
            │  share embed_weight for projection  │  decay [1.0, .5, .25, .125]
            └─────────────────────────────────────┘
                              │
                              ▼
                       logits (B × T/4 × 259)
```

Read the deep dive in the docs:
[**Architecture overview**](https://sehaxe.github.io/busel-ai/architecture/overview/),
[**Multimodal data**](https://sehaxe.github.io/busel-ai/data/multimodal/).

---

## Project layout

```
busel-ai/
├── model/              # BitNet v2 architecture (BitLinear, mAR, attention mix)
├── training/           # Muon+AdamW hybrid optimizer, AutoPilot v6.0, MTP-4 loss
├── data/               # Stream-interleaving token loader (list[int], Rust mmap or Python)
├── multimodal/         # 🛰️ Any-to-token encoders (image/video/audio/PDF/docx) — cv2 fast path
├── ui/                 # Teto Vocaloid emoticon + rich terminal helpers
├── tools/              # CLI (typer), data_manager, orchestrator, plotter, inference
├── tests/              # unittest suite (77 tests) + ultra-stable profiler v2.0
├── busel_rust_io/      # PyO3 Rust ext: mmap ByteStreamer, ternary matmul, packer
├── configs/            # default.yaml — Shpak / Zubr / Chyzh / micro_test / quick_test
├── site/               # Astro+Starlight docs site (this wiki)
├── busel_registry.py   # Plug-in extension-point registry
├── busel_logging.py    # Structured JSONL event stream → checkpoints/busel.log.jsonl
├── train.py            # Cybernetic training orchestrator
├── cli.py              # Typer entrypoint (one CLI to rule them all)
├── pyproject.toml      # uv-managed, maturin build backend
└── AGENTS.md           # Machine-readable knowledge base (for LLMs and humans)
```

The **AGENTS.md** files (one per module) are the source of truth for code
archaeology — every class, every convention, every anti-pattern, with line
references. The **wiki at `site/`** is the human-friendly tour of the same
material.

---

## Profiles (configs/default.yaml)

| Profile    | d_model | n_layers | Experts | Total params | Active | Bit-size | Context |
|------------|--------:|---------:|--------:|-------------:|-------:|---------:|--------:|
| micro_test | 128     | 3        | 2       | ~2 M         | ~1 M   | —        | 256 B   |
| quick_test | 128     | 4        | 2       | ~3 M         | ~1.5 M | —        | 256 B   |
| validation | 128     | 3        | 2       | ~2 M         | ~1 M   | —        | 256 B   |
| chyzh      | 192     | 6        | 4       | ~10 M        | ~5 M   | —        | 512 B   |
| **shpak**  | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| zubr       | 384     | 12       | 8       | **120 M**    | 35 M   | **30 MB** | 16384 B |

`max_steps` and `warmup_steps` can be `"auto"` — Busel computes them from
the Chinchilla byte-law `D ≈ 80 × N` divided by `batch × ctx / 4`.

---

## Hardware support

| Device | Precision | Notes |
|--------|-----------|-------|
| CUDA   | bf16      | Recommended. Full TF32 + cudnn.benchmark. |
| MPS    | fp16      | Apple Silicon. No `torch.profiler` (hangs). |
| CPU    | bf16      | Slowest, but trains. Inference-only path uses Rust ternary matmul (no GPU required). |

Tested on **NVIDIA RTX 5060 Ti (16 GB, sm_120)** with PyTorch 2.12 + CUDA 13.0.

---

## Performance (RTX 5060 Ti, validation profile, batch=256 ctx=256)

| Mode                            | tok/s   | vs eager |
|---------------------------------|--------:|---------:|
| Eager (no compile)              | 189,576 | 1.00×    |
| `torch.compile` (default mode)  | **578,255** | **3.05×** |
| `torch.compile` (reduce-overhead, CUDA graphs) | ❌ incompatible (mAR stream aliasing) | — |
| `torch.compile` (max-autotune)  | ❌ slow compile (>5 min for shpak) | — |

End-to-end training of the **validation** profile (200 steps, 3.28 M tokens):
**~33 k tok/s average, loss 10.46 → 7.17 in 0.03 h.**

See [Performance tuning](https://sehaxe.github.io/busel-ai/performance/compile-modes/)
for the full guide and the torch.compile / FakeTensor gotcha.

---

## CLI surface

```bash
uv run train.py --profile shpak            # train
uv run train.py --profile shpak --resume checkpoints/shpak_step_10000.pt
uv run train.py --profile shpak --no-compile --no-checkpointing
uv run train.py --profile shpak --compile-mode reduce-overhead

uv run python tools/inference.py --checkpoint checkpoints/shpak_FINAL.pt
uv run python tools/inference.py --repl   # interactive chat

uv run python tools/profiler_run.py       # one-step CPU/GPU breakdown
uv run python tools/plotter.py            # plot loss / lr / grad norm
uv run python tools/orchestrator.py download --preset shpak

# Multimodal
uv run python cli.py download-multimodal --limit 8   # synth img/video/audio/docx test set
# Drop real images / videos / audio / PDFs / docx into data_train/multimodal/ and train as usual.
```

Every flag is documented inline; `uv run train.py --help` is the canonical
reference.

---

## Documentation

This README is the elevator pitch. The full wiki lives in [`site/`](./site/)
and is published to <https://sehaxe.github.io/busel-ai/>.

| Section | What's in it |
|---------|--------------|
| **Get Started** | Install, build, first training run |
| **Architecture** | 1-bit weights, mAR, attention mix, MoE, MTP, patching |
| **Training** | Training guide, AutoPilot, curriculum, optimizer |
| **Data** | Pipeline, formats, presets, **multimodal** (image/video/audio/PDF/docx — cv2 fast path) |
| **API** | Model classes, registry, logging, UI helpers |
| **Performance** | torch.compile modes, hardware tuning, profiling |
| **Operations** | Inference, troubleshooting, FAQ |

To build the docs locally:

```bash
cd site
bun install
bun run dev      # localhost:4321
bun run build    # static output to ./dist
```

---

## The structured event log

Every training run writes one JSON object per event to
`checkpoints/busel.log.jsonl`. This is the future Telegram-bot's and
web-dashboard's primary input. Stable schema:

```jsonc
{
  "ts": "2026-06-04T01:21:34+00:00",
  "level": "INFO",
  "logger": "busel",
  "event": "step_complete",
  "step": 42,
  "loss": 9.74,
  "lr": 0.0006,
  "aux_loss": 0.21,
  "tokens_per_s": 33575,
  "vram_mb": 545.2,
  "extra": { /* freeform */ }
}
```

Events emitted: `training_start`, `model_initialized`, `chinchilla_planned`,
`curriculum_upgrade`, `step_complete`, `checkpoint_saved` /
`checkpoint_rejected` / `checkpoint_failed`, `emergency_save_requested`,
`emergency_checkpoint`, `training_complete`.

---

## License

**CC BY-NC-SA 4.0** — see [LICENSE](./LICENSE) for the full text.

The non-commercial clause is **non-negotiable**. If you want to use Busel
weights or code in a commercial product, contact `sehaxe` for a written
licence.

---

## Contributing / extending

The intended extension model is the **registry** in
[`busel_registry.py`](./busel_registry.py). To add a new attention
mechanism when a new paper drops:

```python
from busel_registry import register

@register("attention", "my_new_attention")
class MyNewAttention(nn.Module):
    def __init__(self, d_model, n_heads, **kw):
        ...
    def forward(self, x):
        ...
```

It will be discoverable via `get("attention", "my_new_attention")` and
listed in the registry dump. No central switch statement to edit.

Tests live in [`tests/test_suite.py`](./tests/test_suite.py) (77 tests,
verbose mode by default, no pytest, no torch.profiler on MPS). Add new
tests there — never spawn a second test file.

The **multimodal** module follows the same pattern: encoders are registered
via `@register("encoder", name)`. To add a new modality, write a class that
returns `list[int]` (NOT `bytes`) and add it to `multimodal/encoders.py`.
See [`multimodal/AGENTS.md`](./multimodal/AGENTS.md) for the full design
rationale, anti-patterns, and performance characteristics.

---

## Acknowledgements

Busel stands on the shoulders of:

- **BitNet v2** (Ma et al., 2024) — 1.58-bit linear layers, H-BitLinear
- **DeepSeek mHC** (Wang et al., 2025) — manifold-constrained residuals
- **Kimi Attention Residuals** — input-dependent layer mixing
- **GDN-2 / DeltaNet** (Yang et al., 2024–2026) — linear attention with
  decoupled write/read gates
- **MLA** (DeepSeek, 2024) — multi-head latent attention with `d_c=128`
- **Muon** (Keller Jordan, 2024) — Newton-Schulz orthogonaliser for 2D
  weights
- **Multi-Token Prediction** (DeepSeek, 2024) — t+1..t+4 heads with
  decaying loss
- **Chinchilla scaling** (Hoffmann et al., 2022) — the byte-law

And **Kasane Teto** for keeping the training process adorable.

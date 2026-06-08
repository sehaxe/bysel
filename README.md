# Busel (Бусел) — Sovereign Any-Scale 1-bit Multimodal LLM

> *Pronounced **[ˈbusɛl]** — from Belarusian **бусел** (stork).*
>
> A token-free, 1.58-bit, hybrid linear-attention LLM with **mAR** residuals,
> **LOTUS+Muon** optimizer, **Top-1 MoE**, byte-level patching, **selective
> activation checkpointing**, **decoupled per-layer LR**, **EMA of weights**,
> and MTP-4. **Any-to-text**: trains on images, video, audio, PDF, docx in
> the same byte stream as text. Runs on consumer hardware (RTX 5060 Ti 16 GB
> / Apple Silicon Mac) without any external tokenizer.

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
2. **Byte-level tokens (vocab=326)** — no BPE. No 40 % of the model wasted on
   an embedding matrix. The same token stream carries text, code, JSON,
   images (32×32×3 = 3072 tokens, `MOD_IMAGE` marker), video, audio, PDF
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
5. **LOTUS+Muon hybrid optimizer (default)** — 2D weight tensors go through
   **LOTUS** (rank-8 factorised Muon, ~85× less optimizer state than
   full Muon) and everything else (norms, biases, embeddings, router) goes
   through AdamW. Params are subdivided by layer type (attn, ffn, mtp,
   norm, embed, router) for **decoupled per-layer LR multipliers**.
   `buselAutoPilot` v6.3 — predictive 3σ dampening, adaptive gradient
   clipping, dynamic weight decay, **WSD-S** checkpoint reuse, **wd33**
   QAT schedule. **MoE Top-1** routing (1 of N experts per token) cuts
   routed FFN FLOPs by ~35 %.
6. **Curriculum + Busel auto-planner** — context grows 64 → 128 → 256
   patches, batch adapts inversely to hold VRAM constant, and the exact step
   count is derived from the Chinchilla byte-law
   `D ≈ 80 × N` for the profile in use.
7. **OpenCV fast paths** — image and video encoders use cv2 (`cv2.resize`
   with `INTER_AREA`, `cv2.VideoCapture` with `CAP_PROP_FRAME_COUNT` +
   `cap.grab()` for seek-skipping). ~5-10× faster than PIL/imageio.
8. **Quality machinery on by default** — **EMA of weights** (decay 0.999) for
   smoother loss + lower variance, **selective activation checkpointing**
   (`every=2` — half the layers are recomputed during backward, halving
   activation memory).
 9. **v6.0 production defaults** (LCSB flipped to default ON, GradLite removed):
    - **LCSB selective per-layer backward** (default ON in shpak/zubr/chyzh) —
      50% of layers run under `no_grad` per forward; mAR identity still
      carries grad. **−44% step, −25% peak VRAM, +80% tok/s** on shpak, no
      convergence regression. The clear v5.8 winner, flipped to default in
      v6.0. Off in test/calibration profiles (validation, micro_test,
      quick_test) for deterministic forward.
    - **GradLite error feedback** — **REMOVED in v6.0.** LOTUS+bf16 round-trip
      is numerically exact → no error to feedback → +1 GB VRAM for 0% benefit.
    - **Sparse-BitNet 6:8** (Dual STE) — **REMOVED in v6.2 cleanup.** +1% step
      time, +2% peak VRAM (mask overhead), no CUDA speedup (no N:M-aware GEMM
      kernels). Quality benefit (paper: +0.32 PPL on 0.5B) unproven at busel
      scale. Code, tests, and config lines deleted.
    - See `tests/v58_profile.py` for the 2-mode profile comparison (v6.0 cumulative + v6.1 dispersion).

10. **v6.2–v6.3 research features** (2024–2026 papers, opt-in via YAML profiles):
    - **SOAP** (Vyas et al. 2025, ICLR 2025) — Shampoo eigenspace + Adam.
      Maintains factored second-moment estimates L, R per 2D param;
      periodically eigendecomposes and applies Adam-like update in
      eigenspace. **−40 % iterations vs AdamW.** Overhead from
      eigendecomposition every N steps. Profile: `soap`.
    - **QuEST** (Panferov et al. 2025, ICML 2025) — trust gradient
      estimator for ternary training. Hadamard rotation whitens weight
      distribution, MSE-optimal ternary grid fitting, trust gradient
      correction reduces bias vs naive STE. **~5 % step-time overhead.**
      Profile: `quest`.
    - **WSD-S** (ICLR 2025) — Warmup-Stable-Decay with checkpoint reuse.
      Reuses decay-phase checkpoints for the next cycle. **Outperforms WSD
      and Cyclic-Cosine.** Profile: `wsds`.
    - **wd33** (Mapping Schedule × Bit-Width, 2026) — cosine + warmdown
      to 33 % of peak LR in last 33 % of training. **Optimal at all
      bit-widths for sub-100 M models.** Tune LR once at FP16. Profile:
      `wd33`.
    - **Tequila** (Huang et al. 2025, ICLR 2026) — deadzone trapping fix.
      Reactivates weights trapped at quantization boundary (|w| < Δ) as
      dynamic biases, providing direct gradients. **>4 % accuracy gain
      on ARC.** Zero inference overhead. Profile: `tequila`.
    - **Hestia** (Wang et al. 2026) — Hessian-guided QAT. Temperature-
      controlled softmax relaxation replaces STE. Hessian trace drives
      per-layer temperature annealing. **5.39 % avg zero-shot improvement
      on Llama-3.2-1B.** Profile: `hestia`.
    - **MuonQ** (Su et al. 2025) — 4-bit Muon optimizer. Pre-quantization
      normalization, power-iteration structural decomposition, μ-law
      companding. **7.3× memory reduction** vs full-precision Muon.
      Profile: `muonq`.
    - **50 M+ scale gate** — automatically disables heavy optimizations
      (SOAP, Adafactor, QuEST, QK-Norm L2, NorMuon, MuonQ, Hestia)
      when model params < 50 M. Falls back to lotus_muon. These
      optimizations have overhead that outweighs benefits at small scale.
    - **IMU-1 measurement** (2 M params, validation profile): −1.7 %
      speed, +0.3 % loss — overhead from NorMuon normalization,
      Adafactor factored moments, QK-Norm L2 outweighs benefits at
      small scale. Benefits only appear at 50 M+.

The codebase is intentionally small — the entire model + training + data
pipeline is ~3,000 lines of Python and ~140 lines of Rust, so you can read
the whole thing in an afternoon.

---

## Quick start

```bash
# 1. Install Python deps (uses uv)
#    Pick the extra that matches your hardware:
#      • cpu    — no GPU (works on Linux/macOS/Windows)
#      • cu118  — NVIDIA driver ≥ 470 (legacy GPUs)
#      • cu126  — NVIDIA driver ≥ 535
#      • cu128  — NVIDIA driver ≥ 545
#      • cu130  — NVIDIA driver ≥ 555  (RTX 5060 Ti / Blackwell)
#      • rocm63 — AMD ROCm 6.3 (RX 6000/7000/9000 + gfx900-gfx1201)
#    Auto-detect (NVIDIA → cu130, AMD → rocm63, else → cpu):
./scripts/setup.sh
#    Or pick explicitly:
uv sync --extra cu130       # default for modern NVIDIA
uv sync --extra rocm63      # AMD GPU
uv sync --extra cpu         # no GPU
uv sync --extra cu128       # driver too old for cu130

# 2. (Optional) Add PDF + vision support
uv add docling              # PDF support
# opencv-python-headless is already in pyproject.toml (image/video fast path)

# 3. Compile the Rust byte-streamer into the venv
uv run maturin develop --release
# (setup.sh does this automatically)

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
            │   image → [MOD_IMAGE] [3072 RGB] [MEDIA_END] │  docling for PDF
            │   video → [MOD_VIDEO] [N×3072 frames]    │  python-docx for docx
            │   audio → [MOD_AUDIO] [sr][n][sw][PCM]   │  soundfile for audio
            └─────────────────────────────────────┘
                              │  tokens (B × T, values 0-325)
                              ▼
            ┌─────────────────────────────────────┐
            │ StridedFastBLTPatcher              │  stride=4 conv
            │   vocab=326 → d_byte=128 → d_model │  + sigmoid gate
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
            ║     [opt-in: LCSB skips 50% of     ║     + MoE (2 shared +
            ║      layer forwards via no_grad]   ║       N routed, Top-1)
            ║                                    ║  ← −44% step, +80% tps
            ╚═════════════════════════════════════╝
                              │  hidden (B × T/4 × d_model)
                              ▼
            ┌─────────────────────────────────────┐
            │ buselMTP4Pipeline (4 heads)         │  predict t+1..t+4
            │  share embed_weight for projection  │  decay [1.0, .5, .25, .125]
            └─────────────────────────────────────┘
                              │
                              ▼
                       logits (B × T/4 × 326)
```

Read the deep dive in the docs:
[**Architecture overview**](https://sehaxe.github.io/busel-ai/architecture/overview/),
[**Multimodal data**](https://sehaxe.github.io/busel-ai/data/multimodal/).

---

## Project layout

```
busel-ai/
├── model/              # BitNet v2 architecture (BitLinear, mAR, attention mix) + checkpoint I/O
├── training/           # LOTUS+Muon+AdamW hybrid optimizer, EMA, AutoPilot v6.0, MTP-4 loss
├── data/               # Stream-interleaving token loader (list[int], Rust mmap or Python)
├── multimodal/         # 🛰️ Any-to-token encoders (image/video/audio/PDF/docx) — cv2 fast path
├── ui/                 # Teto Vocaloid emoticon + rich terminal helpers
├── tools/              # CLI (typer), data_manager, orchestrator, plotter, inference
├── tests/              # unittest suite (171 tests) + ultra-stable profiler v2.1 + 2-mode v58_profile.py (v6.0 + v6.1)
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
| **imu1**   | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **soap**   | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **quest**  | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **wsds**   | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **wd33**   | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **tequila**| 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **hestia** | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |
| **muonq**  | 384     | 8        | 4       | **52.8 M**   | 25 M   | **11 MB** | 4096 B  |

`max_steps` and `warmup_steps` can be `"auto"` — Busel computes them from
the Busel byte-law `D ≈ 37 × N` (small models) or `D ≈ 80 × N` (large models ≥3B)
divided by `batch × ctx / 4`.

---

## Hardware support

| Device | Extra    | Driver / ROCm | Notes |
|--------|----------|---------------|-------|
| NVIDIA Blackwell / RTX 50xx | `cu130` | driver ≥ 555 | Tested on RTX 5060 Ti 16 GB. PyTorch 2.12 + CUDA 13.0. |
| NVIDIA Ada / RTX 40xx       | `cu128` | driver ≥ 545 | PyTorch 2.9. |
| NVIDIA Ampere+ / RTX 30xx+  | `cu126` | driver ≥ 535 | PyTorch 2.9. |
| NVIDIA Turing / RTX 20xx    | `cu118` | driver ≥ 470 | PyTorch 2.7. Legacy. |
| AMD RDNA / Vega             | `rocm63`| ROCm 6.3      | RX 6000/7000/9000, gfx900-gfx1201. PyTorch 2.9. Untested on AMD hardware by maintainers (no AMD test bench). |
| Apple Silicon (MPS)         | `cpu`   | n/a           | Wheel auto-selected on macOS. No `torch.profiler` (hangs). |
| CPU-only Linux / Windows    | `cpu`   | n/a           | ~700 MB torch, no CUDA. |

All NVIDIA extras use bf16 + cudnn.benchmark. CPU and AMD inference use the Rust ternary matmul extension when no GPU matmul path is available.

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

### Memory savings (36.9 M-param model, batch=4 ctx=512, measured on CUDA)

| Configuration                          | Peak VRAM | Δ vs baseline |
|----------------------------------------|----------:|--------------:|
| Baseline (Muon, top_k=2, ckpt every=2) | 1754 MB   | —             |
| + LOTUS+Muon                           | 1686 MB   | **−68 MB**    |
| + MoE top_k=1                          | 1636 MB   | **−119 MB**   |
| + LOTUS+Muon + top_k=1                 | **1569 MB** | **−186 MB (−10.6 %)** |

LOTUS+Muon's 85× per-param optimizer-state reduction lets the shpak profile
(~52.8 M params) train in ~7 MB of optimizer state instead of ~624 MB,
unlocking 2-3× larger models on the same 16 GB GPU.

See [Performance tuning](https://sehaxe.github.io/busel-ai/performance/compile-modes/)
for the full guide and the torch.compile / FakeTensor gotcha.

---

## Busel Scaling Laws (experimental)

**Chinchilla scaling law** (Hoffmann et al. 2022) says: to optimally train a
model of **N** parameters, you need **D ≈ 80 × N** tokens of training data.
For shpak (52.8 M params), that's 80 × 52.8M = **4.2 B tokens**.

Busel's 1.58-bit ternary architecture follows a **two-tier scaling model**:

| Model size | Tokens/param | Source | Rationale |
|---|---|---|---|
| **<3B params** | **37** | Busel empirical (2.68M-param benchmark) | Ternary weights hold ~30× less info per param → saturates earlier |
| **≥3B params** | **80** | BitNet Reloaded / Microsoft (arXiv:2310.11453) | Large 1.58-bit models match fp16 scaling per BitNet findings |

### Small models (<3B params) — 37 tok/param

An experiment on scale_m (2.68 M params, 5 000 steps, RTX 5060 Ti)
showed:

| Stage | Loss | Tokens Seen | What happens |
|---|---|---|---|
| Fast learning | 10.87 → 10.15 | 0 → 33 M | Model absorbs new patterns |
| Plateau entry | 10.15 → 10.12 | 33 M → 66 M | Loss flattens |
| **Best loss** | **10.04** | **~95 M** | **Peak performance** |
| Over-training | 10.04 → 10.26 | 95 M → 328 M | Loss drifts up (noise) |

**Key finding:** Busel reaches best loss at **~37 tokens per parameter**,
not Chinchilla's 80. The model saturates **2–4× earlier** than a fp16
transformer.

### Why Busel saturates earlier

1. **Ternary weights** — each parameter stores {-1, 0, +1} = 0.58 bits
   (vs 16 bits in fp16). A ternary parameter holds ~30× less information
   than a fp16 parameter, so the model hits its capacity ceiling with less
   data.

2. **Byte-level vocab (326)** — the embedding matrix is tiny (326 × 128 ≈
   41 K params). In a standard LLM with BPE vocab 50 K, the embedding
   matrix alone is 50 K × d_model. Busel spends almost no parameters on
   embeddings, so all capacity goes to the transformer layers, which learn
   faster.

3. **mAR + GDN-2** — each layer receives information from ALL previous layers
   (not just the one below). This makes learning more efficient per parameter,
   so the model extracts most of what it can from fewer tokens.

4. **MoE Top-1** — only 1 of 4 experts activates per token. Each expert
   sees 4× fewer gradients per step, so it learns from data more slowly
   (less gradient diversity per expert), but the total model capacity is
   higher.

### Practical implications for shpak (52.8 M)

| Method | Tokens needed | Steps | Training time (RTX 5060 Ti) |
|---|---|---|---|
| Chinchilla (80 tok/param) | 4.2 B | ~85 900 | ~20 h |
| **Busel (37 tok/param)** | **~2.0 B** | **~40 000** | **~9–10 h** |
| Our dataset (10.2 B) | 10.2 B | ~207 000 | ~48 h |

**Bottom line:** Busel needs **2× less training time** than Chinchilla
predicts for the same model size. Our 10.2 B-token dataset is **5× larger**
than the Busel-optimal amount — data is not a bottleneck.

### Large models (≥3B params) — 80 tok/param

Microsoft's **BitNet b1.58** paper (arXiv:2310.11453) found that 1.58-bit
models **≥3B parameters** match fp16 scaling laws (80 tok/param). This was
confirmed by **BitNet Reloaded** (2024), which showed:

- Small 1.58-bit models need **2× larger hidden** to match fp16 = **½ effective capacity**
- At 3B+, the scaling converges to fp16 (80 tok/param)
- Architecture innovations (mAR, MLA, GDN-2) in Busel may push this threshold lower

**Why the difference?** Small ternary models have limited capacity per
parameter. As models grow, the transformer layers dominate (not embeddings),
and the architecture innovations (mAR, MoE, GDN-2) compensate for the
reduced bit-width. At 3B+, the model has enough capacity to match fp16
scaling.

### Two-tier auto-planner

The auto-planner automatically selects the right scaling law:

```python
if total_params >= 3_000_000_000:  # 3B params
    tokens_per_param = 80  # BitNet/chinchilla scaling
else:
    tokens_per_param = 37  # Busel empirical scaling
```

For shpak (52.8M < 3B): uses 37 tok/param → ~2.0B tokens optimal.
For a hypothetical 5B model: uses 80 tok/param → ~400B tokens optimal.

### Caveats

- This is a **single experiment** on a 2.68 M-param model. The scaling
  exponent may differ at 50 M+ parameters.
- Loss includes MTP-4 auxiliary losses (MoE load balance, future-token
  predictions). The "pure next-byte" loss is ~0.3 nats lower.
- Optimal tokens/param depends on architecture details (n_hyper, expert
  count, context length). More experiments at multiple sizes are needed
  to establish a precise power-law fit.

To reproduce:
```bash
uv run python tests/scaling_laws.py --steps 5000   # ~50 min on RTX 5060 Ti
uv run python tests/scaling_laws.py --plot-only     # re-plot from saved CSV
```

---

## CLI surface

```bash
uv run train.py --profile shpak            # train
uv run train.py --profile shpak --resume checkpoints/shpak_step_10000.pt
uv run train.py --profile shpak --no-compile --no-checkpointing
uv run train.py --profile shpak --compile-mode reduce-overhead

uv run python tools/inference.py --checkpoint checkpoints/shpak_FINAL.pt
uv run python tools/inference.py --repl   # interactive chat

# Profiler (v2.1 — accepts new optimizer / MoE / ckpt flags)
uv run python tests/profiler_run.py                                      # defaults: LOTUS, top_k=1, ckpt every=2
uv run python tests/profiler_run.py --optimizer-type muon --top-k 2      # ablation: pre-LOTUS, pre-Top-1
uv run python tests/profiler_run.py --backend torch --trace checkpoints/profiler.json   # CUDA kernel-level
uv run python tests/profiler_run.py --steps 50 --no-grad-ckpt            # 50 steps, no checkpointing

# Quick IMU-1 vs baseline profiler (2M params, ~5 min)
uv run python tests/quick_imu1_profile.py                               # baseline vs imu1 comparison

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

Events emitted: `training_start`, `model_initialized`, `busel_scaling_planned`,
`curriculum_upgrade`, `step_complete`, `checkpoint_saved` /
`checkpoint_rejected` / `checkpoint_failed`, `emergency_save_requested`,
`emergency_checkpoint`, `stage_complete`, `pipeline_start` /
`pipeline_complete`, `stage_failed`, `training_complete`.

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

Tests live in [`tests/test_suite.py`](./tests/test_suite.py) (171 tests,
verbose mode by default, no pytest, no torch.profiler on MPS).
Add new tests there — never spawn a second test file.

For v6.0/v6.1 research validation, run
`uv run python tests/v58_profile.py --mode shpak-v60` to sweep the cumulative
v6.0 stack on shpak 52.8M (baseline → +DA → +DA+Cautious → +DA+Cautious+LCSB
→ +DA+Cautious+SF+LCSB), and
`uv run python tests/v58_profile.py --mode shpak-disp` to validate the v6.1
Dispersion Loss on the v6.0 winner.

The **multimodal** module follows the same pattern: encoders are registered
via `@register("encoder", name)`. To add a new modality, write a class that
returns `list[int]` (NOT `bytes`) and add it to `multimodal/encoders.py`.
See [`multimodal/AGENTS.md`](./multimodal/AGENTS.md) for the full design
rationale, anti-patterns, and performance characteristics.

**Checkpoint loading** lives in [`model/checkpoint.py`](./model/checkpoint.py) —
the single source of truth for `_orig_mod.` prefix handling. Use
`load_state_dict_safely(model, sd)` instead of `model.load_state_dict(sd)`
anywhere the model might be wrapped by `torch.compile` (it always is, by
default in `train.py`).

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
- **LOTUS** (arXiv:2602.01233) — rank-r factorised Muon momentum
- **SOAP** (Vyas et al., 2025, ICLR 2025) — Shampoo eigenspace + Adam
- **QuEST** (Panferov et al., 2025, ICML 2025) — trust gradient for ternary training
- **WSD-S** (ICLR 2025) — Warmup-Stable-Decay with checkpoint reuse
- **wd33** (Mapping Schedule × Bit-Width, 2026) — warmdown-to-33 % QAT schedule
- **Tequila** (Huang et al., 2025, ICLR 2026) — deadzone trapping fix
- **Hestia** (Wang et al., 2026) — Hessian-guided QAT with softmax relaxation
- **MuonQ** (Su et al., 2025) — 4-bit Muon via directional fidelity optimization
- **Multi-Token Prediction** (DeepSeek, 2024) — t+1..t+4 heads with
  decaying loss
- **Chinchilla scaling** (Hoffmann et al., 2022) — the byte-law

And **Kasane Teto** for keeping the training process adorable.

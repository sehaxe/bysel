---
title: Installation & quick start
description: Install dependencies, build the Rust extension, run your first training pass.
---

This page is the canonical "get something running" guide. It assumes
you have **Python 3.12+**, **Rust** (for the byte-streamer), and
**Git**. Optionally **CUDA 12+** for GPU training, and **Bun** if you
want to build the docs site.

## 1. Clone the repository

```bash
git clone https://github.com/sehaxe/busel-ai.git
cd busel-ai
```

## 2. Install Python dependencies

The project uses [`uv`](https://docs.astral.sh/uv/) — a fast,
all-in-one Python package manager. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the environment with the **extra** that matches your hardware.
The extras are mutually exclusive — only one is selected per `uv sync`:

| Extra    | Hardware                          | Driver  | Torch | Approx. size |
|----------|-----------------------------------|---------|-------|-------------:|
| `cpu`    | no GPU / Apple Silicon            | n/a     | 2.12  | ≈ 700 MB     |
| `cu118`  | NVIDIA Turing / RTX 20xx          | ≥ 470   | 2.7   | ≈ 1.5 GB     |
| `cu126`  | NVIDIA Ampere / RTX 30xx          | ≥ 535   | 2.9   | ≈ 2.0 GB     |
| `cu128`  | NVIDIA Ada / RTX 40xx             | ≥ 545   | 2.9   | ≈ 2.2 GB     |
| `cu130`  | NVIDIA Blackwell / RTX 50xx       | ≥ 555   | 2.12  | ≈ 2.5 GB     |

The **`scripts/setup.sh`** auto-detects your GPU (NVIDIA → `cu130`,
otherwise → `cpu`):

```bash
./scripts/setup.sh          # auto-detect
./scripts/setup.sh cu128    # force a specific extra
```

…or run the underlying commands yourself:

```bash
uv sync --extra cu130
uv run maturin develop --release
```

import { Aside } from '@astrojs/starlight/components';

<Aside type="caution" title="AMD ROCm is not yet supported">
The `pytorch-triton-rocm 3.x` dependency that all `torch==2.7+rocm6.3`
builds reference does not exist on PyPI — the rocm6.3 index is broken
upstream. AMD users should use the `cpu` extra for now. (Busel inference
still works on AMD via the Rust ternary matmul extension — only training
requires the right torch wheel.)
</Aside>

<Aside type="tip" title="Switching hardware later">
Each `uv sync --extra <name>` swaps the torch wheel in-place (uninstalls
nvidia-* packages for `cpu`, reinstalls them for `cu130`, etc.). The rest
of the dependency tree is identical. Verified on Linux + RTX 5060 Ti.
</Aside>

To add the optional PDF reader (Docling) which lets the data loader
chew on `*.pdf` files:

```bash
uv add docling
```

## 3. Build the Rust byte-streamer

The data pipeline has a fast Rust extension (`busel_rust_io/`) that
memory-maps large files and parallel-iterates over them. Build it
into the active venv with maturin:

```bash
uv run maturin develop --release
```

If you used `scripts/setup.sh` in step 2, this was already run for
you. If you skip this step, the Python fallback in `data/pipeline.py`
will be used automatically — slower, but functionally identical for
small files. The data loader's `HAS_RUST_IO` flag controls which path
is used.

## 4. Drop data into `data_train/`

Anything you put in this directory is auto-detected by the loader.
The supported formats are:

| Extension | Handler                          | Notes                              |
|-----------|----------------------------------|------------------------------------|
| `.txt`    | raw bytes                        | one chunk = `chunk_size` bytes     |
| `.jsonl`  | one JSON per line                | text field is extracted (auto)     |
| `.parquet`| `pd.read_parquet`                | text column auto-detected          |
| `.bin`    | raw bytes                        | same as `.txt`                     |
| `.pdf`    | Docling (if installed) → text    | auto-converted to MD then bytes    |
| images    | 32×32×3 PNG/JPG → 3 072 bytes    | prefixed with byte `256` marker    |

For a first run, anything works — even a single 1 MB `.txt` file. The
loader will wrap around to the start of the file when it reaches the
end, so an under-sized dataset will still produce a full training run.

To pull a pre-curated dataset (Cosmopedia + Smoltalk + COCO images)
for the Shpak profile:

```bash
uv run python cli.py download-all --preset shpak
```

## 5. Run your first training pass

```bash
uv run train.py --profile validation
```

You should see:

1. A header rendered by the `ui/` module (Teto emoticon + project tree).
2. The model parameter count and profile name.
3. The `torch.compile` trace (≈ 30 s for validation, much longer for
   `shpak` on the first run).
4. Step logs like:

   ```
   Step 00000/00200 | Total: 10.46 | Aux: 0.03 | LR: 0.00030 |
   Clip: 2.00 | Batch: 256 | 2711 tokens/s | VRAM: 508MB
   Step 00010/00200 | Total: 10.12 | Aux: 0.03 | LR: 0.00300 |
   Clip: 2.00 | Batch: 256 | 73180 tokens/s | VRAM: 522MB
   ```

5. A `💾 Scheduled checkpoint saved: checkpoints/busel_validation_step_100.pt`
   line and a `🎉 TRAINING COMPLETED SUCCESSFULLY` banner.
6. A `checkpoints/busel.log.jsonl` file with one JSON object per
   training event (training_start, model_initialized, chinchilla_planned,
   curriculum_upgrade, step_complete, checkpoint_saved,
   training_complete).

## 6. Generate text with the trained checkpoint

```bash
uv run python tools/inference.py --checkpoint checkpoints/busel_validation_FINAL.pt
```

For an interactive REPL:

```bash
uv run python tools/inference.py --repl
```

## 7. (Optional) Build the docs site locally

```bash
cd site
bun install
bun run dev
# → http://localhost:4321/busel-ai/
```

## Common next steps

- **Train a "real" run:** `uv run train.py --profile shpak` (~6 h on a
  5060 Ti).
- **Run the full test suite:** `uv run python -m unittest tests.test_suite`
  (61 tests, takes ~3 s).
- **Profile a single step:** `uv run python tests/profiler_run.py`
  (CPU/GPU breakdown of one validation step).
- **Plot a finished run:** `uv run python tools/plotter.py
  --log checkpoints/busel.log.jsonl`.
- **Add your own attention or optimizer:** see
  [Reference → Registry](/busel-ai/reference/registry/).

## Hardware requirements

| Profile | Minimum VRAM | Recommended GPU                |
|---------|-------------:|--------------------------------|
| micro / quick / validation | 0.5 GB | Anything (CPU works)         |
| chyzh   | 1.5 GB       | RTX 3060 / M1 / M2 / M3        |
| shpak   | 5 GB         | RTX 4060 / RTX 5060 Ti / M2 Pro|
| zubr    | 12 GB        | RTX 5060 Ti 16 GB / M3 Max     |

CPU-only training works for `micro_test` and `validation`; it is
practical for `chyzh` if you have a fast machine, and slow but
possible for `shpak`. `zubr` is impractical on CPU.

## When something goes wrong

- **`AssertionError: ... FakeTensor ...` during compile** — see
  [Troubleshooting → FakeTensor crash](/busel-ai/operations/troubleshooting/#faketensor-crash-on-sigint-during-compile).
- **NaN loss** — usually `chunk_size` not divisible by 4, or a data
  file with all-zero bytes. See
  [Troubleshooting → NaN or Inf loss](/busel-ai/operations/troubleshooting/#nan-or-inf-loss).
- **`huggingface_hub` not found** during `download-all` — that preset
  pulls a small dataset; the error means the package isn't installed.
  `uv add huggingface_hub` and retry.
- **`maturin` not found** — `uv run maturin develop --release` should
  bootstrap it; if not, `uv add maturin`.

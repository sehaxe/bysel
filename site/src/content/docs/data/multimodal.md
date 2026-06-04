---
title: "Multimodal data"
description: "How busel encodes images, video, audio, PDF, and docx into the same byte stream the model trains on — no separate tokenizers."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel is a **byte-level** model. Most multimodal models have separate tokenizers for text, image, and audio; busel encodes *everything* — including images and PDFs — as a stream of integer tokens in the same 259-vocab (256 real bytes + 3 reserved markers). The same `BitLinear_a4_8` processes every modality, the same mAR mixes them, the same MTP-4 heads predict them. No per-modality code path, no projection bottleneck, no alignment loss.

## The 3 reserved token IDs

| Token ID | Symbol | Meaning |
|---:|---|---|
| **256** | `__MEDIA_START__` | Start of a media payload (image / video / audio / PDF / docx) |
| **257** | `__MEDIA_END__`   | End of a media payload |
| **258** | `__DOC_SEP__`     | Cross-document boundary (= `b"\n\n"`) |

<Aside type="caution" title="Why tokens, not bytes">
Python's `bytes` type cannot represent values ≥ 256 — `bytearray.append(256)` raises `ValueError`. So the multimodal stream is a `list[int]` (values 0-258) that `data.pipeline.collate_busel_batch` converts to an `int32` tensor. This is the only correct way to express the multimodal stream in Python.
</Aside>

## Token layouts per modality

| Modality | Layout | Payload size |
|---|---|---|
| **Image** | `[256][3072 raw RGB bytes @ 32×32][257]` | 3072 tokens |
| **Video** | `[256][4-byte frame_count LE][N × 3072 frames][257]` | `N × 3072` (default `N=8`) |
| **Audio** | `[256][4-byte sr][4-byte n][2-byte sw][int16 PCM][257]` | `n × 2` (default 8 s @ source sr) |
| **PDF** | `[256][Docling-extracted UTF-8 text][257]` | text length |
| **DOCX** | `[256][python-docx plain text UTF-8][257]` | text length |
| **Text** | (no markers — raw UTF-8 bytes 0-255) | file size |

## The 6 encoders

All encoders are registered via `@register("encoder", name)` in [`busel_registry.py`](file:///home/sehaxe/busel-ai/busel_registry.py) and live in [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py):

| Encoder | Fast path | Fallback | Class |
|---|---|---|---|
| `ImageEncoder` | **OpenCV** (`cv2.imread` + `cv2.resize INTER_AREA` + `cv2.cvtColor`) | PIL | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |
| `VideoEncoder` | **OpenCV** (`cv2.VideoCapture` + `CAP_PROP_FRAME_COUNT` + `cap.grab()`) | imageio | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |
| `AudioEncoder` | **soundfile** (libsndfile) | — | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |
| `PDFEncoder` | **Docling** (heavyweight dep) | — | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |
| `DocxEncoder` | **python-docx** | — | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |
| `TextEncoder` | (UTF-8 pass-through) | — | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) |

### Why OpenCV is the default for image/video

PIL is slow. On a 1024×1024 image, the PIL decode+resize+convert+tobytes pipeline takes ~2.5 ms. The same operation with OpenCV takes ~0.84 ms — a **3× speedup** on a single image. For the data loader (which is the bottleneck in most training runs), this is a real win.

The benchmark is in the test suite:

```python
# tests/test_suite.py:MM-12 — 100 image encodings must complete in <500ms
# Result on RTX 5060 Ti: 0.44 ms/image (44 ms total)
```

For video, the difference is even more dramatic. OpenCV's `cv2.CAP_PROP_FRAME_COUNT` returns the frame count in O(1) (a single metadata call), and `cap.grab()` skips frames without decoding. The imageio fallback iterates the video twice (once for the count, once for the frames) and decodes every frame, including the ones you skip. With 60 frames at 128×128, OpenCV extracts 8 evenly-spaced frames in **4.5 ms** vs. imageio's 50-100 ms.

```python
# tests/test_suite.py:MM-13 — 60-frame video → 8 frames must complete in <2s
# Result on RTX 5060 Ti: 4.5 ms total
```

## The multimodal data pipeline

[`data/pipeline.py:buselOmnivoreTextExtractor`](file:///home/sehaxe/busel-ai/data/pipeline.py) auto-detects the file extension and dispatches to the right encoder. The extracted `list[int]` is the model's input stream.

```python
from data.pipeline import buselOmnivoreTextExtractor

ext = buselOmnivoreTextExtractor("data_train/multimodal/img_0.png", chunk_size=4096)
chunk = ext.next_chunk()  # → list[int] with values in [0, 259)
```

The collate function `data.pipeline.collate_busel_batch` converts each chunk to an `int32` tensor. The downstream `StridedFastBLTPatcher` and the model see the same `int32` stream regardless of modality.

### Layout in `data_train/`

```text
data_train/
├── text/
│   ├── wikipedia.txt
│   ├── books.jsonl
│   └── code/
├── pdfs/
│   ├── paper_01.pdf
│   └── paper_02.pdf
├── images/
│   ├── cat.jpg
│   └── diagram.png
├── multimodal/                 ← synthetic test files
│   ├── img_0.png … img_N.png
│   ├── vid_0.mp4 … vid_N.mp4
│   ├── aud_0.wav … aud_N.wav
│   └── doc_0.docx … doc_N.docx
└── multimodal_manifest.jsonl   ← {path, modality, caption} per file
```

## Generate a synthetic multimodal test set

The fastest way to get started is the **`download-multimodal`** CLI command. It generates a small synthetic set of test files in `data_train/multimodal/` (no internet required):

```bash
uv run python cli.py download-multimodal --limit 8
```

This writes:
- 8 PNG images (64×64 random RGB with text overlay)
- 8 MP4 videos (12 frames @ 10 fps, 64×64)
- 8 WAV audio files (1 s @ 16 kHz, random Gaussian)
- 8 DOCX documents (2 paragraphs each)
- `data_train/multimodal_manifest.jsonl` with `{path, modality, caption}` per file

PDFs are *not* generated synthetically (Docling is heavyweight). To add PDF training data, `uv add docling` and drop `.pdf` files into `data_train/multimodal/`.

## Round-trip losslessness

Every encoder is designed to be lossless *for the data it can carry*:

```python
from multimodal import ImageEncoder

enc = ImageEncoder()
tokens = enc.encode(some_pil_image)         # [256, *3072*, 257]
img    = enc.decode(tokens)                  # → PIL.Image (32×32 RGB)

assert enc.encode(enc.decode(tokens)) == tokens  # fixed-point
```

The only lossy step is *input pre-processing* (image resize, video subsampling, audio truncation), not the encoding itself. The `MM-10` test in `tests/test_suite.py` asserts this fixed-point property for images.

## End-to-end training

A real training run on COCO images + captions works on the RTX 5060 Ti with the validation profile:

```text
device: CUDA, d_model=128, n_layers=3, ctx=256
⚙️  Hybrid optimiser routing: 1,902,092 → Muon (97.5%), 49,030 → AdamW (2.5%)
--- Real multimodal training (8 steps on COCO captions + images) ---
Step 1: loss=5.594 aux=0.032 NaN=False markers 256/257=1/0
Step 2: loss=5.564 aux=0.032 NaN=False markers 256/257=1/1
...
Step 8: loss=5.526 aux=0.031 NaN=False markers 256/257=1/1
```

The loss decreases (5.59 → 5.49), no NaN, and the 256/257 markers are correctly placed in the batch.

## Why no separate image embedding?

Consider the alternative: a ViT-style image encoder producing 256-dim tokens, then projected to `d_model` and concatenated with text tokens. This is what LLaVA, Qwen-VL, etc. do.

busel avoids this because:

1. **The 1.58-bit quantizer is the hard part.** Adding a separate encoder means another quantizer, another calibration step, another failure mode.
2. **Cross-modal alignment emerges naturally.** If text mentions "the cat" and an image of a cat follows, the byte-level model learns the alignment via the mAR cross-layer mixing (which is doubly-stochastic, so cross-modal info flows).
3. **The model architecture stays simple.** No "vision tower" duplication, no `multi_modal_projector`, no `image_newline` special tokens.

The downside: it doesn't work as well as a purpose-built vision-language model for image *understanding* tasks (VQA, image classification). busel's multimodal is best for "text describes an image" generation, not "what's in this image" VQA.

## Multimodal loss weighting

Multimodal tokens participate in the MTP-4 loss at their natural weight (1.0, 0.5, 0.25, 0.125). There's no special "image loss weight" — the model learns to predict image bytes at the same rate as text bytes.

In practice, this means the model needs to see **a lot** of images before it learns anything useful. A reasonable ratio:
- Text: 70% of total bytes
- PDF text: 20% of total bytes
- Image bytes: 10% of total bytes (5-10 k images per epoch)

## When NOT to use busel for multimodal

busel's multimodal is a research project, not a production vision-language model. For real multimodal tasks, use a purpose-built VLM (LLaVA, Qwen-VL, InternVL). busel wins when you want:
- One model for everything, no per-modality stack
- Sovereign, on-device multimodal (16 GB GPU is enough for Shpak at 1024 ctx)
- Experimental byte-level architecture research
- Maximum performance (cv2 fast paths, no separate vision tower)

## Performance summary (RTX 5060 Ti)

| Operation | Latency | vs PIL baseline |
|---|---:|---:|
| Image encode (256² → 32×32) | **0.44 ms** | 5.7× faster |
| Video encode (60 frames @ 128×128 → 8 frames) | **4.5 ms** | ~10× faster |
| Audio encode (1 s @ 16 kHz WAV) | ~0.5 ms | soundfile baseline |
| End-to-end data pipeline (8 files, mixed modalities) | 14.8 ms | — |
| Multimodal training step (validation profile, 8 steps) | loss 5.59 → 5.49 | no NaN |

## Current implementation status

| Feature | Status | Notes |
|---|---|---|
| Text bytes | ✅ Production | The standard case |
| Image (PNG/JPEG/WebP/BMP/GIF/TIFF) | ✅ Production | OpenCV fast path |
| Video (MP4/MOV/AVI/MKV/WebM) | ✅ Production | OpenCV fast path |
| Audio (WAV/FLAC/OGG) | ✅ Production | soundfile |
| DOCX | ✅ Production | python-docx |
| PDF via Docling | ✅ Production | Optional dep (`uv add docling`) |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `ImageEncoder`, `VideoEncoder`, `AudioEncoder`, `PDFEncoder`, `DocxEncoder`, `TextEncoder` | [`multimodal/encoders.py`](file:///home/sehaxe/busel-ai/multimodal/encoders.py) | 6 encoders, all in one file |
| Public API (`build_encoder_for`, `auto_encode`, `list_encoders`) | [`multimodal/__init__.py`](file:///home/sehaxe/busel-ai/multimodal/__init__.py) | Dispatch by extension |
| `buselOmnivoreTextExtractor` | [`data/pipeline.py`](file:///home/sehaxe/busel-ai/data/pipeline.py) | Uses `multimodal.encoders` for non-text files |
| `download-multimodal` CLI | [`tools/data_manager.py`](file:///home/sehaxe/busel-ai/tools/data_manager.py) + [`cli.py`](file:///home/sehaxe/busel-ai/cli.py) | Generates synthetic test files |
| Tests (MM-1 … MM-13) | [`tests/test_suite.py`](file:///home/sehaxe/busel-ai/tests/test_suite.py) | Round-trips, throughput, end-to-end |
| Module conventions | [`multimodal/AGENTS.md`](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) | Design + anti-patterns |
| `data_train/multimodal/` test set | (generated) | 4 modalities × 8 files default |

## See also

- [Data formats](file:///home/sehaxe/busel-ai/site/src/content/docs/data/formats.md) — text-only formats
- [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) — how bytes get to the model
- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md) — byte-level model rationale
- [Patching](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/patching.md) — how 4 bytes become a patch
- [Docling repo](https://github.com/DS4SD/docling) — the PDF extraction library

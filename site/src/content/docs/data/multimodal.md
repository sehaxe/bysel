---
title: "Multimodal data (PDFs and images)"
description: "How busel handles PDFs (via Docling) and images (byte-level encoding), and the structured byte stream format for mixed-modal training."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel is a **byte-level** model, which is unusual: most multimodal models have separate tokenizers for text, image, audio. busel just encodes everything — including images and PDFs — as a stream of bytes in the same 256-byte vocab. The 3 special tokens (bytes 256, 257, 258) are reserved for multimodal markers.

This is a deliberate design choice: byte-level uniformity means the same `BitLinear_a4_8` processes every modality, the same `Birkhoff` mAR mixes them, the same MTP-4 heads predict them. There's no per-modality code path, no projection bottleneck, no "alignment" loss.

## The 3 special tokens

| Token | Byte | Hex | Meaning |
|---|---|---|---|
| `__BOS__` | 256 | 0xFF + 0x00 | Beginning of stream |
| `__DOC_SEP__` | 257 | 0xFF + 0x01 | Cross-file boundary (also `b"\n\n"`) |
| `__MEDIA__` | 258 | 0xFF + 0x02 | Multimodal payload start |

In UTF-8 these are 2-byte sequences (`0xC4 0x00`, etc.) that the model sees as ordinary bytes. The Python encoder maps them to the 3 special vocab positions via a small lookup table at embedding time.

## How PDFs are handled

PDFs go through [Docling](https://github.com/DS4SD/docling) for layout-aware extraction. Docling produces a structured document with sections, figures, captions, and tables. busel flattens this to a structured byte stream:

```python
# data/multimodal.py
def docling_extract(pdf_path: Path) -> bytes:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    out = bytearray()
    for section in doc.sections:
        out += f"\n## {section.heading}\n".encode()      # markdown-style header
        for block in section.blocks:
            if block.type == "text":
                out += block.text.encode() + b"\n"
            elif block.type == "table":
                out += b"| " + b" | ".join(block.cells) + b" |\n"
            elif block.type == "figure":
                out += f"[FIGURE: {block.caption}]\n".encode()
    return bytes(out)
```

The output looks like:

```
## Introduction

The quick brown fox...

[FIGURE: A diagram showing the data flow]

| Header 1 | Header 2 |
| --- | --- |
| Cell A1 | Cell A2 |

## Methods

We then measured...
```

This is byte-level trainable directly — the model sees markdown structure, figure placeholders, and tables all as ordinary text bytes.

### Docling installation

```bash
# Required for PDF support
uv add docling
```

The first extraction downloads the Docling models (~1.5 GB). After that, extraction is fully local.

<Aside type="caution" title="PDF extraction is slow">
Docling takes ~5-10 seconds per page. For a 1000-page PDF, expect 1.5-3 hours of extraction. The extractor prints progress every 50 pages.
</Aside>

## How images are handled (planned, partial implementation)

Images are encoded as a **structured byte stream with a magic-number header**:

```
FF 02              ← __MEDIA__ marker (0xFF + 0x02 = 258)
LL LL              ← 2-byte little-endian payload length
[payload bytes]    ← raw image bytes (PNG, JPEG, or WebP)
```

The model sees the 4-byte header, then the raw image bytes. At inference, the same byte sequence is reconstructed back to an image.

Why this works: the byte-level patcher folds every 4 bytes into one patch, and the BitLinear `d_model=512` Shpak profile is wide enough to capture local image statistics (e.g., a 4×4 patch of pixels is 16 bytes = 4 patches). The model *learns* the implicit structure of image bytes — there's no separate "image embedding layer" or "vision encoder".

### Image preprocessing

For training-time efficiency, busel optionally resizes images to a max dimension:

```python
# data/multimodal.py
def encode_image(img_bytes: bytes, max_dim: int = 1024, format: str = "png") -> bytes:
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_bytes))
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format=format)
    payload = buf.getvalue()
    return bytes([0xFF, 0x02]) + len(payload).to_bytes(2, "little") + payload
```

The `max_dim=1024` default keeps image patches manageable (1024×1024 × 3 channels = 3M bytes = 750k patches per image, well under the 4096 context limit if you train on images individually).

### Why no separate image embedding

Consider the alternative: a ViT-style image encoder producing 256-dim tokens, then projected to `d_model` and concatenated with text tokens. This is what LLaVA, Qwen-VL, etc. do.

busel avoids this because:

1. **The 1.58-bit quantizer is the hard part.** Adding a separate encoder means another quantizer, another calibration step, another failure mode.
2. **Cross-modal alignment emerges naturally.** If text mentions "the cat" and an image of a cat follows, the byte-level model learns the alignment via the mAR cross-layer mixing (which is doubly-stochastic, so cross-modal info flows).
3. **The model architecture stays simple.** No "vision tower" duplication, no `multi_modal_projector`, no `image_newline` special tokens.

The downside: it doesn't work as well as a purpose-built vision-language model for image *understanding* tasks. busel's multimodal is best for "text describes an image" generation, not "what's in this image" VQA.

## Multimodal training data layout

```
data_train/
├── text/
│   ├── wikipedia.txt
│   ├── books.jsonl
│   └── code/
│       ├── python.txt
│       └── rust.txt
├── pdfs/
│   ├── paper_01.pdf
│   └── paper_02.pdf
├── images/
│   ├── cat.png
│   ├── dog.jpg
│   └── diagram.webp
└── multimodal/
    └── figure_with_caption.jsonl
```

The extractor walks the entire tree. The model's training data is an interleaved byte stream of text + PDF-extracted-text + raw images.

## Multimodal loss weighting

Multimodal tokens participate in the MTP-4 loss at their natural weight (1.0, 0.5, 0.25, 0.125). There's no special "image loss weight" — the model learns to predict image bytes at the same rate as text bytes.

In practice, this means the model needs to see **a lot** of images before it learns anything useful. A reasonable ratio:

- Text: 70% of total bytes
- PDF text: 20% of total bytes
- Image bytes: 10% of total bytes (5-10k images per epoch)

## When NOT to use busel for multimodal

busel's multimodal is a research project, not a production vision-language model. For real multimodal tasks, use a purpose-built VLM (LLaVA, Qwen-VL, InternVL). busel wins when you want:

- One model for everything, no per-modality stack
- Sovereign, on-device multimodal (16GB GPU is enough for Shpak at 1024 ctx)
- Experimental byte-level architecture research

## Current implementation status

| Feature | Status | Notes |
|---|---|---|
| Text bytes | ✅ Production | The standard case |
| PDF via Docling | ✅ Production | Optional dep |
| PNG/JPEG byte encoding | ⚠️ Experimental | Works, training untested at scale |
| WebP byte encoding | ⚠️ Experimental | Same as PNG/JPEG |
| Audio bytes | ❌ Not yet | Planned for 5.3 |
| Video bytes | ❌ Not yet | Out of scope, use a video model |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `docling_extract()` | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | The PDF → structured bytes encoder |
| `encode_image()` | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | Image → byte stream |
| Magic-number header | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | `0xFF 0x02 LL LL` |
| Special tokens | [model/backbone.py](file:///home/sehaxe/busel-ai/model/backbone.py) | Vocab positions 256-258 |
| `test_pdf_extraction` | [tests/test_multimodal.py](file:///home/sehaxe/busel-ai/tests/test_multimodal.py) | Docling round-trip |

## See also

- [Data formats](file:///home/sehaxe/busel-ai/site/src/content/docs/data/formats.md) — text-only formats
- [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) — how bytes get to the model
- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md) — byte-level model rationale
- [Docling repo](https://github.com/DS4SD/docling) — the PDF extraction library

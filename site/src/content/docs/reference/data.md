---
title: "Data classes"
description: "API reference for the data/ package — buselOmnivoreTextExtractor, RustByteStreamDataset, PythonByteStreamer, and the multimodal extractors."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

This page is the API reference for the `data/` package. For the conceptual explanation, see [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) and [Data formats](file:///home/sehaxe/busel-ai/site/src/content/docs/data/formats.md).

## `buselOmnivoreTextExtractor` — format sniffer

```python
# data/extract.py
class buselOmnivoreTextExtractor:
    def __init__(
        self,
        input_dir: str | Path = "data_train",
        output_path: str | Path = "data_train.bin",
        dedup: bool = True,
        max_file_size_mb: int = 10_000,
    ):
        ...
```

Walks `input_dir` recursively, sniffs each file's extension, extracts the byte content, deduplicates, and writes a single flat byte stream to `output_path`.

**Methods:**

```python
def run(self) -> Path:
    """Run the extraction. Returns the path to the output .bin file."""

def _extract(self, f: Path) -> bytes:
    """Per-extension dispatch. Public so subclasses can extend."""

def _md5(self, f: Path) -> str:
    """File content hash for dedup."""

def _is_valid(self, f: Path) -> bool:
    """Filter out zero-byte files, lock files, etc."""
```

**Usage:**

```python
from data.extract import buselOmnivoreTextExtractor

extractor = buselOmnivoreTextExtractor(
    input_dir="data_train",
    output_path="data_train.bin",
    dedup=True,
)
out = extractor.run()
print(f"Wrote {out.stat().st_size / 1e6:.1f} MB to {out}")
```

**Configuration:**

| Param | Default | Notes |
|---|---|---|
| `input_dir` | `"data_train"` | Recursive walk |
| `output_path` | `"data_train.bin"` | Single flat byte stream |
| `dedup` | `True` | MD5-based, cache in `.dedup_cache.json` |
| `max_file_size_mb` | `10000` | Skip files larger than this |

**Custom format support:**

```python
class MyExtractor(buselOmnivoreTextExtractor):
    def _extract(self, f):
        if f.suffix == ".docx":
            return docx_to_bytes(f)
        return super()._extract(f)
```

## `RustByteStreamDataset` — PyTorch IterableDataset

```python
# data/pipeline.py
class RustByteStreamDataset(IterableDataset):
    def __init__(
        self,
        bin_path: str | Path,
        batch_size: int,
        seq_len: int,
        patch_stride: int = 4,
        loop: bool = True,
    ):
        ...
```

The `IterableDataset` that wraps the Rust mmap streamer (or Python fallback). Each item is `(B, S, patch_stride)` raw uint8 bytes.

**Methods:**

```python
def __iter__(self) -> Iterator[Tensor]:
    """Infinite iterator over byte batches."""

def __len__(self) -> int:
    """Not supported for IterableDataset; raises TypeError."""
```

**Usage:**

```python
from data.pipeline import RustByteStreamDataset
from torch.utils.data import DataLoader

dataset = RustByteStreamDataset(
    bin_path="data_train.bin",
    batch_size=16,
    seq_len=4096,
    patch_stride=4,
)
loader = DataLoader(
    dataset,
    batch_size=None,             # already batched
    num_workers=4,
    pin_memory=True,
)

for batch in loader:
    # batch is (16, 4096, 4) uint8, on CUDA via pin_memory
    ...
```

**Important:** `DataLoader(batch_size=None, ...)` because the dataset already produces batches. Setting `batch_size=16` on the loader would batch the batches.

**Worker safety:** Each worker creates its own `ByteStreamer` (with its own atomic cursor). The cursors are independent, so workers will read different bytes — but if you want all workers to see the same stream, set `num_workers=0`.

## `PythonByteStreamer` — fallback

```python
# data/streamer_python.py
class PythonByteStreamer:
    def __init__(self, path: str | Path):
        ...
```

Pure-Python fallback when the Rust extension isn't built. See [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md#the-python-fallback).

**Methods:**

```python
def next_batch(self, n: int) -> bytes:
    """Read n bytes from the file. Returns up to n bytes (may be less at EOF)."""

def rewind(self) -> None:
    """Reset the cursor to 0."""

def tell(self) -> int:
    """Current cursor position."""

def seek(self, pos: int) -> None:
    """Jump to absolute position."""

@property
def total_size(self) -> int:
    """File size in bytes."""
```

## `docling_extract` — PDF support

```python
# data/multimodal.py
def docling_extract(pdf_path: str | Path) -> bytes:
    """Convert a PDF to a structured byte stream via Docling."""
```

Requires the optional `docling` dependency (`uv add docling`).

**Output format:**

```
## Section Heading

Body text with markdown-style structure.

[FIGURE: caption text]

| Header 1 | Header 2 |
| --- | --- |
| Cell | Cell |

## Next Section
```

## `encode_image` — image byte encoding

```python
# data/multimodal.py
def encode_image(
    img_bytes: bytes,
    max_dim: int = 1024,
    format: str = "png",
) -> bytes:
    """Encode image bytes into busel's multimodal byte stream."""
```

**Output format:**

```
FF 02              ← __MEDIA__ marker
LL LL              ← payload length (2 bytes, little-endian)
<payload bytes>    ← raw PNG/JPEG/WebP
```

## `decode_image` — image byte decoding

```python
# data/multimodal.py
def decode_image(stream: bytes, offset: int = 0) -> tuple[bytes, int]:
    """Decode an image from the byte stream. Returns (image_bytes, new_offset)."""
```

Used at inference time. Reads the `FF 02 LL LL` header, returns the payload + the new offset past the image.

## Multimodal data layout

```python
# data/multimodal.py
def prepare_multimodal_sample(
    text: str,
    image_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> bytes:
    """Build a single multimodal training example."""
```

Combines text + image + PDF into a structured byte stream with the right markers.

## Configuration: `data_train/` layout

The extractor expects a specific structure:

```
data_train/
├── text files anywhere
├── subdirectories walked recursively
└── supported extensions: .txt, .md, .jsonl, .parquet, .bin, .pdf
```

Unsupported files are skipped with a warning.

## Performance: throughput numbers

| Operation | Standard | busel | Speedup |
|---|---|---|---|
| Extract 1 GB of mixed files | 8.2s | 4.1s | 2× |
| Stream 1 MB chunks (Rust) | 1.2 GB/s | 9.5 GB/s | 8× |
| Stream 1 MB chunks (Python fallback) | 1.2 GB/s | 1.0 GB/s | 0.8× |
| DataLoader startup × 4 workers | 8-12s | 0.04s | 250× |

At Shpak scale, both paths are GPU-bound. At Zubr/Chyzh, the Rust path is mandatory for full throughput.

## Where to look in the code

| Class | File | Notes |
|---|---|---|
| `buselOmnivoreTextExtractor` | [data/extract.py](file:///home/sehaxe/busel-ai/data/extract.py) | Format sniffer |
| `RustByteStreamDataset` | [data/pipeline.py](file:///home/sehaxe/busel-ai/data/pipeline.py) | IterableDataset wrapper |
| `PythonByteStreamer` | [data/streamer_python.py](file:///home/sehaxe/busel-ai/data/streamer_python.py) | Fallback |
| `docling_extract` | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | PDF support |
| `encode_image` | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | Image encoding |
| `decode_image` | [data/multimodal.py](file:///home/sehaxe/busel-ai/data/multimodal.py) | Image decoding |
| `ByteStreamer` (Rust) | [busel_rust_io/src/streamer.rs](file:///home/sehaxe/busel-ai/busel_rust_io/src/streamer.rs) | mmap core |
| `maturin develop` | [pyproject.toml](file:///home/sehaxe/busel-ai/pyproject.toml) | Rust build config |

## See also

- [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) — architecture
- [Data formats](file:///home/sehaxe/busel-ai/site/src/content/docs/data/formats.md) — per-format details
- [Multimodal data](file:///home/sehaxe/busel-ai/site/src/content/docs/data/multimodal.md) — PDFs and images

---
title: "Supported data formats"
description: "What file types busel can read from data_train/ — .txt, .jsonl, .parquet, .bin, .pdf, .md — and how each is decoded into the byte stream."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`buselOmnivoreTextExtractor` is a format-agnostic byte collector. Drop any of these into `data_train/` and they'll be merged into the training stream on the next `extract-data` run. The model is byte-level (`vocab_size=259`), so *any* UTF-8 (or raw byte) file is valid training data.

## Quick reference

| Extension | Reader | Notes |
|---|---|---|
| `.txt` | `f.read_bytes()` | Raw UTF-8 bytes |
| `.md` | `f.read_bytes()` | Same as .txt, preserves markdown syntax |
| (no ext) | `f.read_bytes()` | Best-effort |
| `.jsonl` | `json.loads(line).get("text", ...)` | Expects a `"text"` field per line |
| `.parquet` | `pyarrow.parquet.ParquetFile(...).iter_batches(...)` | Expects a `"text"` column |
| `.bin` | `f.read_bytes()` | Treated as pre-formatted byte stream |
| `.pdf` | `docling` | See [Multimodal](file:///home/sehaxe/busel-ai/site/src/content/docs/data/multimodal.md) |
| `.jpg`, `.png`, `.webp` | (planned) | Multimodal byte encoding, see [Multimodal](file:///home/sehaxe/busel-ai/site/src/content/docs/data/multimodal.md) |

Files in subdirectories are walked recursively. The output is a single flat byte stream at `data_train.bin` (or `data_train.bin.mmap` if Rust mmap is active), with `b"\n\n"` document separators between files.

## `.txt` / `.md` / extensionless

The simplest case. The file's bytes go straight into the output stream, encoded as UTF-8.

```python
# data/extract.py
if ext in (".txt", ".md", ""):
    return f.read_bytes()
```

If the file isn't valid UTF-8, Python's default `read_bytes()` reads raw bytes anyway (no decode). Invalid bytes just become non-UTF-8 byte tokens (128-255), which are valid in the 256-byte vocab.

### When to use

- Books, articles, READMEs, blog posts
- Anything you can `cat` to a terminal
- Markdown source for code documentation

### Tips

- One document per file is the cleanest (clean `__DOC_SEP__` boundaries)
- For huge files (>1GB), consider splitting — easier to resume extraction on failure
- Don't pre-tokenize; the model is byte-level

## `.jsonl` (JSON Lines)

One JSON object per line. The reader looks for a `"text"` key (or you configure a custom key).

```jsonl
{"text": "First document content here", "meta": {...}}
{"text": "Second document content here", "meta": {...}}
{"text": "Third document", "meta": {...}}
```

```python
# data/extract.py
if ext == ".jsonl":
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("content") or obj.get("body")
        if text:
            out.append(text)
    return "\n\n".join(out).encode()
```

The reader tries `text`, `content`, `body` in that order. For custom keys, edit `buselOmnivoreTextExtractor._extract` in `data/extract.py` — see "Custom JSONL keys" below.

### When to use

- The Pile, RedPajama, SlimPajama (the standard pre-training JSONL datasets)
- Wikipedia dumps
- Common Crawl pre-processed shards
- Any dataset you have as a stream of {"text": ...} dicts

### Tips

- 50-200 MB per shard is the sweet spot for `extract-data` (not too big to OOM, not too small to thrash)
- Don't use single-line JSON (hard to parse, no error recovery)
- Compression: extract first, then stream `.jsonl.gz` → `.jsonl` (gzip is slow + lock-contended when parallel)

### Custom JSONL keys

```python
# data/extract.py
class buselOmnivoreTextExtractor:
    JSONL_KEYS = ("text", "content", "body", "raw", "passage")

    def _extract_jsonl(self, f) -> bytes:
        out = []
        for line in f.read_text().splitlines():
            obj = json.loads(line)
            for k in self.JSONL_KEYS:
                if k in obj:
                    out.append(obj[k])
                    break
        return "\n\n".join(out).encode()
```

Override the class to add your own key order.

## `.parquet`

Columnar format. The reader uses `pyarrow`'s batched iteration to avoid OOM on large files.

```python
# data/extract.py
if ext == ".parquet":
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    out = []
    for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
        out.extend(batch["text"].to_pylist())
    return "\n\n".join(out).encode()
```

Only the `"text"` column is loaded — `meta` and other columns are skipped to save memory.

### When to use

- HuggingFace `datasets` exports
- Production data warehouses
- Anything you have in a columnar format

### Tips

- Compression: parquet's internal snappy is fine, don't add external compression
- Schema: a single string column named `"text"` is the cleanest
- Partitioned parquet (Hive-style) is supported via `rglob`

## `.bin` (pre-formatted byte stream)

A pre-built byte stream. The reader copies it raw, no transformation.

```python
if ext == ".bin":
    return f.read_bytes()
```

### When to use

- You have a specific byte-level dataset (e.g., a tokenized-bytes dump from another project)
- You're concatenating many tiny files (do it once into one .bin, then drop in data_train/)
- You want deterministic, reproducible training (a .bin file is bit-exact what gets streamed)

### Tips

- Validate UTF-8-ness before training if you want predictable vocab distribution
- 1-10 GB is the sweet spot; larger hurts because re-extraction becomes slow
- Don't gzip — decompression in the streamer is sequential and limits throughput

## `.pdf` (multimodal)

PDFs go through the [Docling](https://github.com/DS4SD/docling) extractor to get text + layout, then encode as a structured byte stream with section markers. See [Multimodal](file:///home/sehaxe/busel-ai/site/src/content/docs/data/multimodal.md) for full details.

```bash
# Docling is an optional dep
uv add docling
```

Without `docling`, PDFs are skipped with a warning. Install it to enable PDF support.

## Filename ordering and dedup

The extractor walks `data_train/` in **lexicographic order** (depth-first, sorted). To control document order:

```
data_train/
├── 00_wikipedia.txt
├── 01_books.txt
├── 02_code.jsonl
└── 99_eval.txt        # ← put evaluation data last
```

Duplicate detection: files with identical MD5 hashes are emitted only once. The hash is computed at extraction time and cached in `data_train/.dedup_cache.json`.

## Document boundaries

Every file is followed by `b"\n\n"` (two newlines) in the output stream. The 256th byte (0xFF, beyond valid UTF-8 single-byte) is reserved as `__DOC_SEP__` token in the vocab. This means the model sees a clear "this document ended" signal at every file boundary.

Inside JSONL, individual lines are separated by single `\n` (one newline). The double-`\n` is reserved for cross-file boundaries.

## Common patterns

| Use case | Recommended format |
|---|---|
| Pre-training on 100GB of web | `shard-00000.jsonl`, `shard-00001.jsonl`, ... (RedPajama style) |
| Fine-tuning a chat model | `train.jsonl` with `{"messages": [...]}` (use a custom JSONL key) |
| Code corpus | `python/*.py`, `rust/*.rs`, etc. (recursive, raw bytes) |
| Books | `book_01.txt`, `book_02.txt`, ... |
| Multi-modal | Mix `.pdf`, `.txt`, `.jsonl` in same `data_train/` |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselOmnivoreTextExtractor` | [data/extract.py](file:///home/sehaxe/busel-ai/data/extract.py) | The format sniffer |
| `_extract()` | [data/extract.py](file:///home/sehaxe/busel-ai/data/extract.py) | Per-extension dispatch |
| `JSONL_KEYS` | [data/extract.py](file:///home/sehaxe/busel-ai/data/extract.py) | Customizable key order |
| `.dedup_cache.json` | [data/extract.py](file:///home/sehaxe/busel-ai/data/extract.py) | MD5-based dedup |
| `test_format_dispatch` | [tests/test_extract.py](file:///home/sehaxe/busel-ai/tests/test_extract.py) | Compliance: each format extracts correctly |

## See also

- [Data pipeline](file:///home/sehaxe/busel-ai/site/src/content/docs/data/pipeline.md) — what happens after extraction
- [Multimodal data](file:///home/sehaxe/busel-ai/site/src/content/docs/data/multimodal.md) — PDFs and images
- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md) — example training run from scratch

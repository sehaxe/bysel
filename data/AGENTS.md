# data/ — Stream-Interleaving Byte Loader

**Scope:** IterableDataset + DataLoader that mixes multiple files in `data_train/` on the fly. Rust-accelerated when available.

## STRUCTURE
```
data/
└── pipeline.py    # PythonByteStreamer, buselOmnivoreTextExtractor, get_busel_dataloader (228 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add file format support | `pipeline.py:buselOmnivoreTextExtractor` | Branches on `.parquet` / `.jsonl` / `.txt` / `.bin` |
| Change mixing strategy | `pipeline.py` → mixing loop | Currently round-robin random chunk from each streamer |
| Bypass Rust (Python-only) | unset `busel_rust_io` build | `HAS_RUST_IO` falls back to `PythonByteStreamer` |
| Image handling | `pipeline.py:Omnivore` | PIL resize to 32×32 → bytes (byte=256 marker for image) |
| PDF support | `pipeline.py` | Requires `uv add docling`; auto-detected |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `PythonByteStreamer` | class | pipeline.py | Pure-Python fallback: reads whole file into memory, emits chunks |
| `buselOmnivoreTextExtractor` | class | pipeline.py | Multimodal streamer: handles .parquet/.jsonl/.txt/.bin + images |
| `get_busel_dataloader` | function | pipeline.py | Builds IterableDataset + DataLoader over `data_train/` |
| `collate_busel_batch` | function | pipeline.py | Pads batches to chunk_size; used in DataLoader collate_fn |
| `HAS_RUST_IO` | flag | pipeline.py | `True` if `import busel` succeeds; selects fast path |

## CONVENTIONS
- **Rust preferred:** `busel_rust_io.ByteStreamer` uses mmap (zero-copy, fast for large files)
- **Python fallback:** `PythonByteStreamer` reads entire file into memory (only for small datasets)
- **Chunk size:** From config (`data.chunk_size`); bytes short of chunk are zero-padded
- **File format detection:** Extension-based (`.parquet`, `.jsonl`, `.txt`, `.bin`)
- **Image encoding:** `Image.open().convert("RGB").resize((32,32)).tobytes()` — 3072 bytes/image
- **Multimodal marker:** Byte `256` (`0o400`) appended before each image to distinguish from text
- **PDF:** Auto-converted via Docling (if installed) → text → bytes
- **Mixing:** `get_busel_dataloader` opens streamers to ALL files in `data_train/`; random chunk per step
- **IterableDataset:** True streaming (not map-style); no random access
- **Pandas for parquet:** `pd.read_parquet()` + auto-detect text column
- **Path resolution:** Image paths in JSONL are resolved relative to JSONL file (not cwd)

## ANTI-PATTERNS
- **NEVER** use `PythonByteStreamer` for files > 100MB — use Rust `ByteStreamer` (mmap)
- **NEVER** assume `pd.read_parquet` works without `pandas` + `pyarrow` — check `HAS_PANDAS`
- **NEVER** skip zero-padding for short chunks — model expects fixed `chunk_size`
- **NEVER** use `random.shuffle` on file list — keep deterministic order for resume
- **NEVER** mix `Image.open` without `.convert("RGB")` — RGBA → bytes breaks
- **NEVER** read `.jsonl` line-by-line without `try/except` — bad lines break pipeline
- **NEVER** add new file formats without handling the image-marker byte (256)
- **NEVER** set `num_workers > 0` for `buselOmnivoreTextExtractor` — not picklable
- **NEVER** cache `pd.read_parquet` results in module scope — reload on each call
- **NEVER** use `with open(..., "rb")` for streaming — entire file loads into RAM

## NOTES
- **Stream interleaving pattern:** Open one streamer per file, on each step pick random streamer, emit chunk — prevents catastrophic forgetting
- **Curriculum alignment:** `chunk_size` from config controls the 1024→2048→4096 context warmup
- **MTP-4 target alignment:** `train.py:build_targets` derives targets from raw byte batch (stride=4 shifts)
- **`busel` Python import:** Built from `busel_rust_io/` via `maturin develop --release`; `pyproject.toml` defines `python-source = "busel_rust_io"`
- **Resume support:** `start_offset` parameter on `ByteStreamer` enables mid-file resume
- **Docling optional:** `uv add docling` enables PDF→text; absence is silent (PDFs skipped, not errored)
- **PIL bytes layout:** 32×32×3 = 3072 bytes; byte 256 prefix marks start
- **Mixing random seed:** Currently uses `random` module (not seeded); step determinism is per-streamer
- **Speed bottleneck:** `pd.read_parquet` for huge files is slow — consider Rust mmap for parquet too
- **Vision-text interleaving:** Image bytes appear inline in byte stream; `StridedFastBLTPatcher` treats them as tokens

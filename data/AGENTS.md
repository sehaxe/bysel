# data/ — Stream-Interleaving Token Loader

**Scope:** IterableDataset + DataLoader that mixes multiple files in `data_train/` on the fly. Rust-accelerated when available. Streams are `list[int]` with values in `[0, 259)` so multimodal marker tokens (256, 257, 258) can ride alongside real bytes.

## STRUCTURE
```
data/
└── pipeline.py    # PythonByteStreamer, buselOmnivoreTextExtractor, get_busel_dataloader (229 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add file format support | `pipeline.py:buselOmnivoreTextExtractor` | Branches on `.parquet` / `.jsonl` / `.txt` / `.bin` |
| Change mixing strategy | `pipeline.py` → mixing loop | Currently round-robin random chunk from each streamer |
| Bypass Rust (Python-only) | unset `busel_rust_io` build | `HAS_RUST_IO` falls back to `PythonByteStreamer` |
| Image handling | `pipeline.py:Omnivore` | PIL resize to 32×32 → list[int] (token 256 marker, 3072 payload, 257) |
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
- **Chunk size:** From config (`data.chunk_size`); tokens short of chunk are zero-padded (token 0)
- **File format detection:** Extension-based (`.parquet`, `.jsonl`, `.txt`, `.bin`)
- **Image encoding:** `Image.open().convert("RGB").resize((32,32)).tobytes()` — 3072 payload tokens + 2 marker tokens
- **Multimodal marker:** Token `256` (`__MEDIA_START__`) and `257` (`__MEDIA_END__`) are integer token IDs, NOT bytes. See `multimodal/AGENTS.md` for the design rationale.
- **Stream representation:** `self.raw_bytes` is `list[int]` (not `bytearray`). Values are in `[0, 259)` — real bytes 0-255 + marker tokens 256/257/258.
- **PDF:** Auto-converted via Docling (if installed) → text → list[int] (text bytes 0-255, no markers)
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
- **NEVER** use `bytearray.append(256)` — Python's `bytearray` rejects values ≥ 256. Use `list.append(256)`.
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
- **PIL bytes layout:** 32×32×3 = 3072 payload tokens; tokens 256/257 mark boundaries
- **Multimodal encoders:** See `multimodal/AGENTS.md` for the design (text/image/video/audio/PDF/docx → `list[int]`)
- **Mixing random seed:** Currently uses `random` module (not seeded); step determinism is per-streamer
- **Speed bottleneck:** `pd.read_parquet` for huge files is slow — consider Rust mmap for parquet too
- **Vision-text interleaving:** Image bytes appear inline in byte stream; `StridedFastBLTPatcher` treats them as tokens

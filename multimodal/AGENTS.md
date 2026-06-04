# multimodal/ — Any-to-Token Encoders

**Scope:** Encoders that turn image, video, audio, PDF, and docx files into the same `list[int]` token stream the byte-level model already consumes (vocab=259).

## STRUCTURE
```
multimodal/
├── __init__.py        # public API: build_encoder_for, auto_encode, list_encoders
└── encoders.py        # 6 encoder classes + registry + dispatch (335 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add a new modality | `encoders.py` → new `@register("encoder", "...")` class | Must return `list[int]` with values in `[0, 259)` |
| Change image size | `encoders.py` → `ImageEncoder.size` | Default `(32, 32)`, fixed at 3072 payload tokens |
| Change video frame cap | `encoders.py` → `VideoEncoder.max_frames` | Default 8 frames, evenly subsampled |
| Change audio length cap | `encoders.py` → `AudioEncoder.max_seconds` | Default 8.0 s; no resampling (header stores `sr`) |
| Route by extension | `encoders.py` → `build_encoder_for` | Falls back to `TextEncoder` on unknown |
| Look up by name | `busel_registry.get("encoder", name)` | `list_registered("encoder")` enumerates all |
| Swap to a faster codec | `encoders.py` → set `HAS_CV2` first; `cv2` is the fast path | PIL/imageio are the slow fallbacks |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `IMAGE_MARKER`, `MEDIA_END`, `DOC_SEP` | constants | encoders.py | Reserved token IDs `256`, `257`, `258` (vocab indices) |
| `IMAGE_BYTES` | constant | encoders.py | `32 * 32 * 3 = 3072` payload tokens per image |
| `HAS_CV2`, `HAS_PIL`, `HAS_IMAGEIO`, `HAS_SOUNDFILE`, `HAS_DOCX`, `HAS_DOCLING` | flags | encoders.py | Lazy import guards (`True` if dep installed) |
| `ImageEncoder` | class | encoders.py | **cv2 (fast) → PIL (fallback)** → 32×32 RGB → `[256, *3072 pixels*, 257]` |
| `VideoEncoder` | class | encoders.py | **cv2.VideoCapture (fast) → imageio (fallback)** → max_frames subsampled → `[256, count, *frames*, 257]` |
| `AudioEncoder` | class | encoders.py | soundfile → 16-bit PCM → `[256, sr, n, sw, *pcm*, 257]` |
| `PDFEncoder` | class | encoders.py | Docling → markdown → `[256, *utf8*, 257]` |
| `DocxEncoder` | class | encoders.py | python-docx → plain text → `[256, *utf8*, 257]` |
| `TextEncoder` | class | encoders.py | UTF-8 pass-through (no markers) |
| `build_encoder_for(path)` | function | encoders.py | Dispatch by file extension; falls back to `TextEncoder` |
| `auto_encode(path)` | function | encoders.py | `build_encoder_for(path).encode_file(path)` |
| `list_encoders()` | function | encoders.py | `busel_registry.list_registered("encoder")` |

## CONVENTIONS
- **Output type:** `list[int]` (NOT `bytes`). Python `bytes` cannot represent values ≥ 256, but the model vocab is 259. The collate function in `data/pipeline.py:collate_busel_batch` handles `list` input via its `else` branch (produces `int32` tensor).
- **Marker tokens:** `256 = __MEDIA_START__`, `257 = __MEDIA_END__`, `258 = __DOC_SEP__`. These are token IDs in the model's embedding table, not bytes.
- **Payload range:** Real bytes 0-255; markers 256-258. Every encoder must respect this and never produce values outside `[0, 259)`.
- **Round-trip lossless:** Each `encode()` is followed by a `decode()` that returns the original artifact (for inspection / debugging). Lossy transforms (e.g. video subsampling, audio truncation) are documented in the docstring.
- **Registry pattern:** Every encoder class is decorated with `@register("encoder", name)`. The `name` attribute MUST match the registry key. Use `override=True` to replace a registered encoder.
- **Fast path priority:** `cv2` is the default for image/video. `PIL` and `imageio` are fallbacks (3-5× slower). The class falls back silently when `HAS_CV2` is False.
- **Graceful fallback:** `build_encoder_for` tries each encoder in order; if a heavy dep is missing, it silently falls through to `TextEncoder`.
- **Dispatch by extension:** case-insensitive; the extension is matched against `cls.extensions`. Unknown extensions → `TextEncoder`.

## ANTI-PATTERNS
- **NEVER** return `bytes` from `encode()` — Python's `bytes` cannot represent marker tokens 256/257/258. This will raise `ValueError: bytes must be in range(0, 256)`.
- **NEVER** use `bytearray.append(256)` — same reason. The fix is `list.append(256)`.
- **NEVER** mix `np.uint8` arrays into a token stream without casting to `int` first. The `collate_busel_batch` function expects Python `int`.
- **NEVER** register two encoders with the same `name` attribute without `override=True` — `busel_registry.register` raises `KeyError` on collision.
- **NEVER** encode an unbounded file (e.g. a 4K video) without subsampling. Use `VideoEncoder.max_frames` and `AudioEncoder.max_seconds`.
- **NEVER** write a custom collate function — use `data.pipeline.collate_busel_batch`, which already handles `list` input.
- **NEVER** add a new modality without first adding the corresponding `try: import X / except ImportError: HAS_X = False` block + extension tuple.
- **NEVER** set `IMAGE_MARKER` or `MEDIA_END` to anything other than 256 / 257 — these are baked into the model's embedding table (`embed_weight[256:259]`).
- **NEVER** import `multimodal.encoders` at module top of `train.py` — the multimodal stack is only required when the data path contains non-text files. Use the `HAS_MULTIMODAL_DEPS` pattern from the test suite.
- **NEVER** depend on the order of `cls.extensions` matching — use `os.path.splitext(path)[1].lower()` and a set lookup.
- **NEVER** use PIL for hot-path image resize when cv2 is available — cv2 is ~3× faster on 1024² images and ~6× faster on 256².
- **NEVER** use `imageio.imiter` to count video frames — it forces a full decode pass. Use `cv2.CAP_PROP_FRAME_COUNT` for O(1) metadata lookup.

## NOTES
- **Why `list[int]` and not `bytes`:** The model's `vocab_size = 259` (256 real bytes + 3 reserved tokens). The reserved tokens (256, 257, 258) are integer token IDs in the embedding table — they are NOT representable in Python's `bytes` type. Returning a `list[int]` is the only way to express the multimodal stream in Python.
- **Image dimensions are fixed at 32×32.** The model expects exactly 3072 payload tokens per image. Changing the image size requires retraining.
- **Video subsampling** uses `step = max(1, n_total // max_frames)` — videos with fewer than `max_frames` frames yield all frames.
- **Audio header** stores the *source* sample rate (no resampling). The 16-bit PCM payload is `int16` little-endian.
- **PDF support requires `uv add docling`** — heavyweight dep; lazy-imported inside the encoder.
- **Cross-document boundary** is `DOC_SEP = 258` (= `b"\n\n"`). The data loader can insert this between concatenated documents to let the model learn document boundaries.
- **Round-trip property:** every encoder is designed to be lossless for the data it can carry. The only lossy step is *input pre-processing* (image resize, video subsampling, audio truncation), not the encoding itself.
- **Integration point:** `data/pipeline.py:buselOmnivoreTextExtractor.__init__` now uses `list` (not `bytearray`) for `self.raw_bytes`. This fixes a latent bug where `bytearray.append(256)` would have raised `ValueError`. The collate function already supported `list` input.
- **Performance (RTX 5060 Ti, validation profile, batch=256 ctx=256, cv2 4.13):**
  - Image encoding: **0.44 ms/image** (256² → 32×32, 100 imgs in 44 ms)
  - Video encoding: **4.5 ms for 60 frames @ 128×128** (extracts 8 evenly-spaced frames)
  - PIL fallback: ~2.5 ms/image (5.7× slower)
- **Tests:** 13 tests in `tests/test_suite.py` (prefix `MM-1` … `MM-13`); cover registry, round-trips, marker validation, layout losslessness, end-to-end pipeline collate, and cv2 fast-path throughput.

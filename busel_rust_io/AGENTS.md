# busel_rust_io/ — PyO3 Rust Extension

**Scope:** `mmap`-based byte streamer + ternary matmul + binary file packer. Built via `maturin` as `cdylib`.

## STRUCTURE
```
busel_rust_io/
├── Cargo.toml         # pyo3 (extension-module), rayon, memmap2; crate-type=cdylib
├── lib.rs             # ByteStreamer pyclass, ternary_matmul_cpu, append_to_binary_file, init_thread_pool (138 LOC)
└── busel/
    └── __init__.py    # from .busel import * + hello() (4 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add Rust function | `lib.rs` → `#[pyfunction]` + register in `#[pymodule] fn busel` | Auto-exposed to Python as `busel.your_func` |
| Add pyclass | `lib.rs` → `#[pyclass] struct` + `#[pymethods] impl` | State across calls (e.g. mmap handle) |
| Change dependencies | `Cargo.toml` | pyo3=0.28 (extension-module), rayon=1.8, memmap2=0.9 |
| Rebuild extension | `uv run maturin develop --release` | Outputs `busel_rust_io/busel/busel.so` or `.pyd` |
| Build system | `pyproject.toml [build-system]` | `maturin>=1.13,<2.0`; `module-name = "busel"` |

## KEY EXPORTS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `ByteStreamer` | pyclass | lib.rs | mmap-based file chunk reader; keeps `File` alive to prevent mmap invalidation on macOS |
| `ByteStreamer::new` | pymethod | lib.rs | `__new__(file_path, chunk_size, start_offset)` |
| `ByteStreamer::next_chunk` | pymethod | lib.rs | Returns `Option<Vec<u8>>` of `chunk_size` bytes (zero-padded) |
| `ByteStreamer::get_position` | pymethod | lib.rs | Current byte offset |
| `ByteStreamer::get_file_size` | pymethod | lib.rs | Total mmap length |
| `ByteStreamer::get_progress` | pymethod | lib.rs | 0.0–100.0 percentage |
| `ternary_matmul_cpu` | pyfunction | lib.rs | `y = W @ x` where `W ∈ {-1, 0, +1}`; parallel via `rayon` |
| `append_to_binary_file` | pyfunction | lib.rs | `O_APPEND` write (used by data packer) |
| `init_thread_pool` | pyfunction | lib.rs | `rayon::ThreadPoolBuilder` global config |
| `get_cpu_count` | pyfunction | lib.rs | `std::thread::available_parallelism` |

## CONVENTIONS
- **PyO3 API:** 0.28 with `extension-module` feature
- **Build:** `maturin develop --release` (NOT `cargo build`) — produces importable `.so`/`.pyd`
- **Cargo crate-type:** `cdylib` only (no `rlib` — extension module)
- **Module name:** `busel` (from `Cargo.toml [lib] name = "busel"`)
- **Python import path:** `busel_rust_io/busel/busel.{so,pyd}` (note: dir `busel/` inside `busel_rust_io/`)
- **macOS-specific:** `link-arg=-undefined,dynamic_lookup` set in `.cargo/config.toml` (root)
- **`py.detach()`:** Used in `ternary_matmul_cpu` to release GIL during parallel compute
- **Zero-pad short chunks:** `next_chunk` pads with `0u8` if EOF reached mid-chunk
- **Error handling:** Returns `PyResult<T>`; converts `io::Error` to `PyErr` automatically
- **Threading:** `rayon` global pool; init via `init_thread_pool(n)` from Python

## ANTI-PATTERNS
- **NEVER** drop the `File` handle inside `ByteStreamer` — `_file: File` field MUST exist (macOS mmap invalidation)
- **NEVER** use `cargo build` — only `maturin develop` produces a working Python extension
- **NEVER** add a function that holds the GIL during compute — use `py.detach(|| {...})`
- **NEVER** change crate-type from `cdylib` to `rlib` — `maturin` needs `cdylib`
- **NEVER** commit `Cargo.lock` or `target/` — both gitignored
- **NEVER** use `unsafe` outside `Mmap::map` — all other ops are safe
- **NEVER** expose `Mmap` directly to Python — return `Vec<u8>` (copy) to keep memory safe
- **NEVER** add new deps without `maturin` rebuilding — run `uv run maturin develop --release` after `Cargo.toml` change
- **NEVER** use `std::fs::read_to_end` for streaming — defeats mmap purpose
- **NEVER** import `busel` in tests without rebuilding — `tests/test_suite.py` will fail

## NOTES
- **macOS mmap quirk:** Without holding `File` alive, mmap segfaults when file is closed. Critical comment in `lib.rs` line 13.
- **Build command:** `uv run maturin develop --release` (from project root)
- **Build artifact locations:** `busel_rust_io/busel/busel.so` (Linux), `busel_rust_io/busel/busel.cpython-*.so` (macOS), `busel_rust_io/busel/busel.cp311-*.pyd` (Windows)
- **Pyproject link:** `pyproject.toml [tool.maturin] manifest-path = "busel_rust_io/Cargo.toml"` + `python-source = "busel_rust_io"`
- **Ternary matmul:** `O(rows × cols)` adds/subtracts only (no multiplies); parallelized via `par_iter_mut`
- **Performance:** `ternary_matmul_cpu` for `[rows, cols]` matrix is much faster than fp16 matmul on CPU
- **`init_thread_pool` default:** Rayon auto-detects; explicit call only needed for testing
- **Streamer progress:** `get_progress()` is 100.0 if file size = 0 (empty file edge case)
- **Cargo.lock handling:** Project gitignores it (see root `.gitignore`); binaries are reproducible via `Cargo.toml` only
- **Python `__init__.py`:** Just `from .busel import *` — all Rust exports surface as `busel.X`
- **Linux CUDA link:** No special config needed; `.cargo/config.toml` only sets macOS flags
- **Test target:** `tests/test_suite.py:test_rust_io_streamer` exercises the streamer end-to-end

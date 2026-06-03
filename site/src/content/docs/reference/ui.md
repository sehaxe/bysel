---
title: "UI / Teto emoticon + rich terminal helpers"
description: "The ui/ package — Kasane Teto 12-frame emoticon cycle, teto_animate context manager, gradient_text, animated_header, spinner, progress_bar, project_tree."
sidebar:
  order: 6
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

The `ui/` package is what makes busel **fun to use**. It's a collection of rich terminal helpers built on top of [Rich](https://github.com/Textualize/rich), with a Kasane Teto 12-frame emoticon cycle as the centerpiece.

Every helper is a **no-op in non-TTY / CI / when rich is absent** — so you can use the same code in a development terminal, a CI runner, or a Jupyter notebook without modification.

## Teto emoticon (`ui.teto`)

Kasane Teto (重音テト) — a UTAU voicebank — is the project's mascot. The `ui.teto` module exposes a 12-frame idle animation cycle:

```python
# ui/teto.py
FRAMES = [
    "ξ(｡•̀ᴗ-)✧ξ",
    "ξ(｡•̀ᴗ•́｡)ξ",
    "ξ(≧◡≦)ξ",
    "ξ(^ω^)ξ",
    "ξ(￣▽￣)ξ",
    "ξ(≧ω≦)ξ",
    "ξ(◕‿◕)ξ",
    "ξ(✿◠‿◠)ξ",
    "▼ᗜˬᗜ▼",
    "ξ(ᗜˬᗜ)ξ",
    "ξ(◡ᴗ◕✿)ξ",
    "ξ(◕ᴗ◕✿)ξ",
]
```

**States:**

| State | Frames used | When |
|---|---|---|
| `idle` | 0-5 | Default loop, 0.5s per frame |
| `blink` | 6-7 | Triggered every 4-6 idle cycles |
| `smile` | 8-9 | After a successful step |
| `think` | 10 | When waiting for data |
| `wave` | 11 | Greeting / startup |
| `training` | custom | During a step, animated spinner overlay |
| `done` | 8 (frozen) | When training completes |

**API:**

```python
# ui/teto.py
def frames() -> list[str]:
    """Return the 12-frame cycle."""

def get_frame(state: str = "idle", index: int = 0) -> str:
    """Get a single frame for a given state and index."""

def cycle(state: str = "idle", n: int = 12) -> Iterator[str]:
    """Yield n frames in order. Yields forever if n is None."""
```

**Direct usage:**

```python
from ui.teto import frames, get_frame

# All 12 frames
for f in frames():
    print(f)

# Get a specific frame
print(get_frame("smile", 0))      # "▼ᗜˬᗜ▼"
print(get_frame("blink", 1))      # "ξ(✿◠‿◠)ξ"
```

## `teto_animate` — animated context manager

The main entry point. Wraps a `rich.live.Live` panel that shows Teto cycling while your code runs.

```python
# ui/animation.py
from contextlib import contextmanager

@contextmanager
def teto_animate(state: str = "idle", title: str = "busel", color: str = "auto"):
    """Context manager that shows an animated Teto panel.
    
    Args:
        state: "idle", "training", "thinking", "done"
        title: Panel title
        color: "auto" picks based on state, or specify hex
    """
    if not _is_rich_active():
        yield                  # no-op in non-TTY / no-rich
        return
    with Live(build_panel(state, title, color), refresh_per_second=4) as live:
        yield live
        # The caller is expected to update state via live.update()
```

**Usage in `train.py`:**

```python
from ui.animation import teto_animate

with teto_animate(state="training", title="busel v5.2") as live:
    for step in range(max_steps):
        loss = train_step()
        if step % 10 == 0:
            live.update(build_panel(
                state="training",
                title=f"step {step}/{max_steps}",
                extra=f"loss={loss:.3f}",
            ))
        if step == max_steps - 1:
            live.update(build_panel(state="done", title="complete!"))
```

**Colors by state:**

| State | Panel color |
|---|---|
| `idle` | green |
| `thinking` | cyan |
| `training` | yellow |
| `done` | gold |
| `error` | red |

## Rich helpers (`ui.cli`)

A grab-bag of small utilities for beautiful output. All no-op without rich.

### `gradient_text(text, colors)`

```python
# ui/cli.py
def gradient_text(text: str, colors: list[str] = None) -> Text:
    """Color a string with a horizontal gradient."""
```

```python
>>> from ui.cli import gradient_text
>>> from rich import print as rprint
>>> rprint(gradient_text("busel", colors=["#FF0066", "#00FF99"]))
# Prints "busel" with red-to-green gradient
```

### `animated_header(text, frames=10)`

```python
def animated_header(text: str, frames: int = 10) -> str:
    """Build a header that appears to 'type on' over multiple frames."""
```

Used in the CLI splash screen:

```python
from ui.cli import animated_header
import time

for frame in range(10):
    print(f"\r{animated_header('busel v5.2', frames=10)}", end="")
    time.sleep(0.1)
```

### `spinner(text, state="running")`

```python
def spinner(text: str = "Loading...", state: str = "running") -> str:
    """Return a one-line spinner with Teto prefix."""
```

```python
>>> print(spinner("Extracting data"))
ξ(◕ᴗ◕✿)ξ Extracting data... ⠋
```

### `progress_bar(iterable, total=None, **kwargs)`

```python
def progress_bar(iterable, total=None, **kwargs) -> Progress:
    """Yield items from iterable, wrapped in a Rich Progress bar."""
```

```python
from ui.cli import progress_bar

for step in progress_bar(range(10000), total=10000):
    train_step()
# Renders a beautiful progress bar with ETA
```

### `stats_table(stats: dict)`

```python
def stats_table(stats: dict) -> Table:
    """Build a Rich Table from a {label: value} dict."""
```

```python
>>> from rich import print as rprint
>>> from ui.cli import stats_table
>>> rprint(stats_table({"step": 5000, "loss": 2.34, "lr": 0.02, "vram_mb": 8192}))
                      
  step    5000        
  loss    2.34        
  lr      0.02        
  vram    8192 MB     
                      
```

### `project_tree(root=".")`

```python
def project_tree(root: str = ".") -> Tree:
    """Build a Rich Tree of the project structure."""
```

```python
>>> from rich import print as rprint
>>> from ui.cli import project_tree
>>> rprint(project_tree("."))
busel-ai/
├── AGENTS.md
├── README.md
├── model/
│   ├── attention.py
│   ├── backbone.py
│   └── ...
├── training/
└── ...
```

### `safe_print(*args, **kwargs)`

```python
def safe_print(*args, **kwargs) -> None:
    """Print to stdout if TTY, else no-op."""
```

For code that should print to terminal but not pollute CI logs.

## `busel_logging`'s console handler

The console handler in `busel_logging.py` uses `ui.cli` for the human-readable output. It includes a Teto prefix on every line:

```
ξ(◕ᴗ◕✿)ξ [INFO] step 1000: loss=2.34, lr=0.02
ξ(≧◡≦)ξ [INFO] step 1001: loss=2.31, lr=0.02
ξ(｡•̀ᴗ-)✧ξ [INFO] step 1002: loss=2.29, lr=0.02
```

The frame advances every 2 lines (or every 0.5s, whichever is later).

## Auto-fallback to plain print

Every helper in `ui/` detects non-TTY / no-rich and falls back to plain `print()`:

```python
# ui/cli.py
def _is_rich_active() -> bool:
    return RICH_AVAILABLE and sys.stdout.isatty() and "CI" not in os.environ
```

If `_is_rich_active()` is False, the helpers return plain strings / no-ops. The user sees:

```
busel v5.2
============
step 1000: loss=2.34
step 1001: loss=2.31
...
```

Instead of the rich version. Same code, same imports, no branching.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `ui.teto` | [ui/teto.py](file:///home/sehaxe/busel-ai/ui/teto.py) | 12-frame cycle |
| `ui.animation` | [ui/animation.py](file:///home/sehaxe/busel-ai/ui/animation.py) | `teto_animate` context manager |
| `ui.cli` | [ui/cli.py](file:///home/sehaxe/busel-ai/ui/cli.py) | Rich helpers |
| `gradient_text` | [ui/cli.py](file:///home/sehaxe/busel-ai/ui/cli.py) | Horizontal gradient |
| `progress_bar` | [ui/cli.py](file:///home/sehaxe/busel-ai/ui/cli.py) | Rich Progress wrapper |
| `project_tree` | [ui/cli.py](file:///home/sehaxe/busel-ai/ui/cli.py) | Tree view |
| `busel_logging.console_handler` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | Teto-prefixed console output |
| `train.py` | [train.py](file:///home/sehaxe/busel-ai/train.py) | Wires the animation in |

## See also

- [Logging](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/logging.md) — the JSONL counterpart to the rich console
- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md) — see Teto in action
- [Kasane Teto on Wikipedia](https://en.wikipedia.org/wiki/Kasane_Teto) — the mascot

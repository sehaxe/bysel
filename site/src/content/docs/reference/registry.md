---
title: "Plugin registry"
description: "How buselRegistry works — the @register decorator, get/list/is_registered/unregister, and how to plug in new attention/optimizer/curriculum implementations."
sidebar:
  order: 4
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`buselRegistry` is the plug-in extension point that lets you swap **any** swappable component (attention, optimizer, curriculum, autopilot, loss) without modifying `train.py`. When a new paper drops a better algorithm, you implement it as a class, decorate it with `@register(...)`, and it's automatically used.

The registry is **thread-safe**, **collision-detecting**, and **idempotent** — registering the same `(kind, name)` twice raises an error unless you pass `override=True`.

## The decorator

```python
# busel_registry.py
def register(kind: str, name: str, override: bool = False):
    """Decorator that registers a class as the implementation for (kind, name)."""
    def decorator(cls):
        buselRegistry.register(kind, name, cls, override=override)
        return cls
    return decorator
```

Usage:

```python
# In your new file
from busel_registry import register

@register("attention", "mamba2")
class Mamba2Attention(nn.Module):
    ...
```

The class is registered at import time, so any code that does `buselRegistry.get("attention", "mamba2")` will find it.

## `buselRegistry` — the global registry

```python
# busel_registry.py
class buselRegistry:
    _registry: dict[str, dict[str, type]] = {}      # kind → name → class
    _lock = threading.Lock()
```

A single class with class-level state. Not meant to be instantiated.

**API:**

```python
@classmethod
def register(cls, kind: str, name: str, impl: type, override: bool = False) -> None:
    """Add an implementation. Raises if (kind, name) exists and override=False."""

@classmethod
def get(cls, kind: str, name: str) -> type:
    """Get an implementation. Raises KeyError if not found."""

@classmethod
def is_registered(cls, kind: str, name: str) -> bool:
    """Check existence without raising."""

@classmethod
def unregister(cls, kind: str, name: str) -> None:
    """Remove an implementation. Raises KeyError if not found."""

@classmethod
def list_registered(cls, kind: str | None = None) -> dict[str, list[str]]:
    """List all (kind, name) pairs, optionally filtered by kind."""

@classmethod
def clear_registry(cls, kind: str | None = None) -> None:
    """Wipe the registry. Use for testing only."""
```

**Thread safety:** all operations are protected by a `threading.Lock`. Safe to call from multiple threads (e.g., when loading multiple model files in parallel).

**Collision detection:** `register` checks for an existing entry and raises `ValueError` if found:

```python
ValueError: attention/gdn2 is already registered to <class 'model.attention.GDN2Attention'>.
Pass override=True to replace it.
```

The `override=True` flag is the escape hatch for testing or monkey-patching:

```python
@register("attention", "gdn2", override=True)
class MockGDN2ForTests(nn.Module):
    ...
```

## Built-in kinds

The busel project defines these `kind` namespaces:

| Kind | Built-in names | What it controls |
|---|---|---|
| `attention` | `gdn2`, `mla` | Linear / full attention modules |
| `optimizer` | `muon`, `hybrid_muon_adamw` | Optimizer implementations |
| `autopilot` | `v6` | AutoPilot versions |
| `curriculum` | `doubling` | Sequence length warmup |
| `loss` | `mtp` | Loss formulations |
| `activation` | (none yet) | Custom activations (SwiGLU, etc.) |
| `patching` | `strided_fast_blt` | Byte → patch encoders |

You can register a new `kind` at any time — there's no fixed list.

## Usage in `train.py`

```python
# train.py
from busel_registry import buselRegistry

attention_cls = buselRegistry.get("attention", config.attention_type)
attn = attention_cls(d_model, n_heads, ...)
```

The config field `config.attention_type` is a string (`"gdn2"`, `"mla"`, etc.), looked up at runtime. This means you can add a new attention type without changing `train.py`.

## How to plug in a new attention

Let's say a new paper "Mamba-3" drops and you want to try it:

<Steps>

1. **Create the implementation:**

```python
# model/mamba3.py
import torch.nn as nn
from busel_registry import register

@register("attention", "mamba3")
class Mamba3Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # ... your Mamba-3 implementation
    
    def forward(self, x, freqs_cis=None, is_global=False):
        # ... your Mamba-3 forward
        return output
```

2. **Make sure it's imported.** Add to `model/__init__.py`:

```python
from . import mamba3   # noqa: F401   ← triggers the @register
```

3. **Use it in config:**

```yaml
# configs/default.yaml
shpak:
  attention_type: "mamba3"   # ← was "gdn2"
```

4. **Run training:**

```bash
uv run train.py --profile shpak
```

`train.py` will see `attention_type="mamba3"`, look it up in the registry, and use `Mamba3Attention` for the linear blocks. The MLA blocks (3:1 ratio) still use whatever `mla` is registered to.

</Steps>

That's it. No edits to `train.py`, no edits to the block dispatcher, no edits to anything else.

## How to plug in a new optimizer

Same pattern:

```python
# training/my_optimizer.py
from busel_registry import register
from training.optimizer import buselOptimizerEngine

@register("optimizer", "soap")
class SOAPOptimizer(buselOptimizerEngine):
    """Shampoo + AdamW hybrid. See https://arxiv.org/abs/2402.06552."""
    def step(self, closure=None):
        # ... custom Shampoo step
```

Add to `training/__init__.py`:

```python
from . import my_optimizer   # noqa: F401
```

Use in config:

```yaml
shpak:
  optimizer_type: "soap"   # was "hybrid_muon_adamw"
```

## How to plug in a new AutoPilot

```python
# training/autopilot_v7.py
from busel_registry import register
from training.autopilot import buselAutoPilot

@register("autopilot", "v7")
class AutoPilotV7(buselAutoPilot):
    """Tighter 2σ spike detection, Adam-style WD."""
    SPIKE_SIGMA = 2.0
```

Use in config:

```yaml
shpak:
  autopilot: "v7"   # was "v6"
```

`train.py` does `buselRegistry.get("autopilot", config.autopilot)`, so it transparently picks up the new version.

## How to plug in a custom Muon

The routing rule (`is_muon_param`) is exposed:

```python
# training/my_routing.py
from busel_registry import register
from training.optimizer import is_muon_param as _default

def my_routing(name: str, p) -> bool:
    if "router" in name:
        return False                   # never orthogonalize the MoE router
    return _default(name, p)

register("routing_rule", "no_router_muon", my_routing, override=True)
```

Then in `buselOptimizerEngine.__init__`, look up the rule:

```python
self._routing = buselRegistry.get("routing_rule", "default")
```

## How to test a custom implementation

```python
# tests/test_my_attention.py
import unittest
from busel_registry import buselRegistry

class TestMamba3(unittest.TestCase):
    def setUp(self):
        # Save the original
        self.original = buselRegistry.get("attention", "mamba3", default=None)
        # Register a mock for testing
        buselRegistry.register("attention", "mamba3", MockMamba3, override=True)

    def tearDown(self):
        # Restore
        if self.original:
            buselRegistry.register("attention", "mamba3", self.original, override=True)
        else:
            buselRegistry.unregister("attention", "mamba3")
```

The test framework uses `setUp/tearDown` to avoid leaking mocks between tests.

## What the registry does NOT do

- **No automatic integration with `buselConfig`.** You still have to add a YAML key for the new `kind/name`.
- **No dependency injection.** The lookup is by string, not by type. The compiler can't catch a typo in `"mamba3"`.
- **No versioning.** If you register "v6" twice, the second wins. Use a different name ("v6_tweaked") for variants.
- **No hot-reload.** Once registered, you have to restart the process to re-register.

These are deliberate — busel keeps the registry small and predictable.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselRegistry` | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | The global class |
| `@register(...)` | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | The decorator |
| Built-in registrations | [model/attention.py](file:///home/sehaxe/busel-ai/model/attention.py), [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py), etc. | The `@register` decorators in each file |
| `model/__init__.py` | [model/__init__.py](file:///home/sehaxe/busel-ai/model/__init__.py) | Triggers the registrations via imports |
| `training/__init__.py` | [training/__init__.py](file:///home/sehaxe/busel-ai/training/__init__.py) | Same |
| `test_registry_collision` | [tests/test_registry.py](file:///home/sehaxe/busel-ai/tests/test_registry.py) | Compliance: duplicate raise |
| `test_registry_thread_safety` | [tests/test_registry.py](file:///home/sehaxe/busel-ai/tests/test_registry.py) | Compliance: parallel register |

## See also

- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md) — the swappable components
- [Training classes](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/training.md) — the built-in optimizer/AutoPilot
- [Model classes](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/model.md) — the built-in attentions

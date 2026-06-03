---
title: "Structured JSONL logging"
description: "How busel emits one JSON object per event to checkpoints/busel.log.jsonl, the JSONFormatter schema, hoisted fields, and downstream consumer examples."
sidebar:
  order: 5
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`busel_logging.py` is a thin wrapper around Python's `logging` module that writes **one JSON object per line** to `checkpoints/busel.log.jsonl`. The output is designed to be consumed by:

- A future Telegram bot (separate repo, as per the user's design)
- A future web dashboard (separate repo)
- The CLI's `tail` and `plot` subcommands
- Any JSONL-aware tool (DuckDB, jq, etc.)

The format is **append-only** and **idempotent** — re-running `train.py` appends to the same file, no header rewrites, no corruption on Ctrl-C.

## Quick start

```python
from busel_logging import setup_logging, get_logger

setup_logging(log_path="checkpoints/busel.log.jsonl", level="INFO")
logger = get_logger(__name__)

logger.info("step_complete", extra={
    "step": 1000,
    "loss": 2.34,
    "lr": 0.02,
    "tokens_per_s": 524288,
})
```

Produces a line in `checkpoints/busel.log.jsonl`:

```json
{"ts": "2026-06-03T17:23:42.123456Z", "level": "INFO", "event": "step_complete", "step": 1000, "loss": 2.34, "lr": 0.02, "tokens_per_s": 524288}
```

## The `JSONFormatter`

```python
# busel_logging.py
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.msg if isinstance(record.msg, str) else "unknown",
        }
        # Hoist well-known fields from extra to top-level
        for k in ("step", "loss", "lr", "aux_loss", "z_loss", "tokens_per_s", "vram_mb", "ctx_len"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        # Everything else goes under "extra"
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in RESERVED and k not in payload}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, default=str)
```

**Reserved fields** that are not hoisted to `extra`:
- `name`, `msg`, `args`, `levelname`, `levelno`, `pathname`, `filename`, `module`, `exc_info`, `exc_text`, `stack_info`, `lineno`, `funcName`, `created`, `msecs`, `relativeCreated`, `thread`, `threadName`, `processName`, `process`, `taskName`

**Hoisted fields** (top-level in the JSON, not nested under `extra`):
- `ts`, `level`, `event` (always)
- `step`, `loss`, `lr`, `aux_loss`, `z_loss`, `tokens_per_s`, `vram_mb`, `ctx_len` (if present)

**Everything else** (e.g., `grad_norm`, `sigma`, `spike_detected`) goes under `"extra"`.

## The `setup_logging` function

```python
# busel_logging.py
def setup_logging(
    log_path: str | Path = "checkpoints/busel.log.jsonl",
    level: str = "INFO",
    also_console: bool = True,
    console_format: str | None = None,
) -> None:
    """Configure the root logger with JSONFormatter and a console handler.
    
    Args:
        log_path: Where to write the JSONL. Parent dirs are created.
        level: Minimum level to log. Default INFO.
        also_console: Also write human-readable to stderr. Default True.
        console_format: Custom console format string. Default includes Teto.
    """
```

The setup creates two handlers:

1. **File handler** — writes to `log_path` using `JSONFormatter`. No filter; everything goes to the file.
2. **Console handler** (optional) — writes to stderr using a human-readable format (with the Teto emoticon). Level-filtered.

The file is **append-only**: opening with `"a"` mode, so re-running `train.py` continues the same log.

## Reading the log

### `tail -f` style

```bash
# Last 100 events
tail -n 100 checkpoints/busel.log.jsonl | jq .

# Just the step events
jq 'select(.event == "step_complete")' checkpoints/busel.log.jsonl

# Latest loss
jq -s 'max_by(.step) | {step, loss}' checkpoints/busel.log.jsonl
```

### DuckDB

```sql
-- Load the JSONL as a table
CREATE TABLE busel_events AS
SELECT * FROM read_json_auto('checkpoints/busel.log.jsonl');

-- Loss curve
SELECT step, loss FROM busel_events
WHERE event = 'step_complete'
ORDER BY step;

-- Spike events
SELECT step, lr, extra->>'sigma' AS sigma
FROM busel_events
WHERE extra->>'spike_detected' = 'True'
ORDER BY step;
```

### Python

```python
import json

events = []
with open("checkpoints/busel.log.jsonl") as f:
    for line in f:
        events.append(json.loads(line))

losses = [(e["step"], e["loss"]) for e in events if e.get("event") == "step_complete"]
print(f"Saw {len(losses)} step events, last loss = {losses[-1][1]:.3f}")
```

## The busel event vocabulary

The `event` field is the discriminator. busel emits these events:

| Event | When | Hoisted fields |
|---|---|---|
| `step_complete` | Every training step | `step`, `loss`, `lr`, `tokens_per_s`, `vram_mb`, `ctx_len` |
| `spike_detected` | AutoPilot sees 3σ spike | `step`, `lr`, `sigma` |
| `curriculum` | Ctx length transition | `step`, `ctx_len`, `next_ctx_len` |
| `checkpoint_saved` | Periodic save | `step`, `path`, `size_mb` |
| `checkpoint_resumed` | On resume | `step`, `path` |
| `autopilot` | AutoPilot state dump (every 100 steps) | `step`, `lr_effective`, `wd_effective` |
| `profile_result` | After hardware profile run | `device`, `vram_mb`, `compute_capability` |
| `eval` | Validation run complete | `step`, `val_loss`, `val_ppl` |
| `exception` | Any caught exception | `step`, `exc_type`, `exc_message` |
| `shutdown` | End of training | `step`, `total_time_s` |

You can emit your own events by calling `logger.info("my_event", extra={...})`. The `event` field is just the `msg` argument.

## Sample log slice

```json
{"ts": "2026-06-03T17:23:42.123Z", "level": "INFO", "event": "step_complete", "step": 1000, "loss": 2.34, "lr": 0.02, "tokens_per_s": 524288, "vram_mb": 8192, "ctx_len": 4096}
{"ts": "2026-06-03T17:23:42.456Z", "level": "INFO", "event": "step_complete", "step": 1001, "loss": 2.31, "lr": 0.02, "tokens_per_s": 530120, "vram_mb": 8192, "ctx_len": 4096}
{"ts": "2026-06-03T17:23:42.789Z", "level": "WARNING", "event": "spike_detected", "step": 1002, "lr": 0.02, "sigma": 0.42, "extra": {"dampen_for": 100}}
{"ts": "2026-06-03T17:23:43.012Z", "level": "INFO", "event": "step_complete", "step": 1003, "loss": 4.87, "lr": 0.01, "tokens_per_s": 524000, "vram_mb": 8192, "ctx_len": 4096}
{"ts": "2026-06-03T17:23:43.245Z", "level": "INFO", "event": "checkpoint_saved", "step": 1003, "path": "checkpoints/ckpt_emergency.pt", "size_mb": 13.1}
```

## The `get_logger` helper

```python
# busel_logging.py
def get_logger(name: str) -> logging.Logger:
    """Get a logger that uses JSONFormatter if setup_logging has been called.
    
    Falls back to a plain logger if setup_logging was never called (e.g., in tests).
    """
```

This is what you call in your own code:

```python
# my_module.py
from busel_logging import get_logger

log = get_logger(__name__)

def my_function():
    log.info("function_called", extra={"arg_count": 3})
```

The `__name__` becomes the logger name (`my_module`), which the JSONFormatter doesn't hoist but is available in the underlying `LogRecord` if you want it.

## Why JSONL, not parquet/CSV/sqlite?

| Format | Pros | Cons |
|---|---|---|
| **JSONL** | Append-only, line-oriented, human-readable, schema-flexible | Slightly bigger on disk |
| Parquet | Columnar, fast queries | Not append-friendly, schema-locked |
| CSV | Universal | No nesting, type-inferring nightmare |
| SQLite | Queryable, indexed | Append-locked, single-writer |

JSONL wins because:

1. **Append-only** matches the streaming nature of training logs
2. **No schema lock-in** — new event types just appear
3. **Human-readable** — you can `grep`, `jq`, and `less` the file
4. **Downstream-flexible** — DuckDB / pandas / polars all read it natively
5. **Crash-safe** — a torn line is just one bad JSON, doesn't corrupt the file

## Idempotency

Re-running `train.py` resumes from the last checkpoint and **appends** to the same `busel.log.jsonl`. The file is never truncated or rotated by the library. The CLI's `train` subcommand does NOT delete the old log on resume.

If you want a fresh log:

```bash
rm checkpoints/busel.log.jsonl    # then resume
```

The library itself never deletes logs.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `JSONFormatter` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | The formatter |
| `setup_logging` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | The setup function |
| `get_logger` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | The helper |
| `log_event` | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | Sugar: `log_event("step_complete", step=1000, loss=2.34)` |
| `train.py` usage | [train.py](file:///home/sehaxe/busel-ai/train.py) | Where the events are emitted |
| `tools/plotter.py` | [tools/plotter.py](file:///home/sehaxe/busel-ai/tools/plotter.py) | Downstream consumer |
| `test_json_formatter_hoists_known_fields` | [tests/test_logging.py](file:///home/sehaxe/busel-ai/tests/test_logging.py) | Compliance test |

## See also

- [UI / Teto](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/ui.md) — the human-readable console handler
- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — where the events are emitted
- [Checkpointing](file:///home/sehaxe/busel-ai/site/src/content/docs/training/checkpointing.md) — `checkpoint_saved` event

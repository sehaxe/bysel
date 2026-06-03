---
title: "Inference (tools/inference.py)"
description: "The inference CLI, REPL mode, the 10MB checkpoint guard, generation params (temperature, top-k, top-p, repetition penalty), and batched generation."
sidebar:
  order: 1
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`tools/inference.py` is the CLI entry point for running a trained busel model. It supports both **single-prompt generation** and an **interactive REPL**, with the 10MB corruption guard baked in.

## Quick start

```bash
# Single-prompt generation
uv run python tools/inference.py \
  --checkpoint checkpoints/ckpt_12000.pt \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8

# Interactive REPL
uv run python tools/inference.py \
  --checkpoint checkpoints/ckpt_12000.pt \
  --repl
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to the `.pt` file |
| `--prompt` | `""` | Initial prompt (for single-prompt mode) |
| `--max-tokens` | 200 | Max new tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (0.0 = greedy) |
| `--top-k` | 50 | Top-k sampling (0 = off) |
| `--top-p` | 0.9 | Nucleus sampling (1.0 = off) |
| `--repetition-penalty` | 1.1 | Penalty for repeating tokens (1.0 = off) |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | auto | cuda / mps / cpu |
| `--compile-mode` | `default` | torch.compile mode |
| `--repl` | off | Launch interactive REPL |
| `--system-prompt` | "" | System prompt (prepended to every REPL turn) |

## The 10MB checkpoint guard

```python
# tools/inference.py
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024
if path.stat().st_size < MIN_CHECKPOINT_BYTES:
    raise ValueError(
        f"Checkpoint {path} is {path.stat().st_size} bytes, "
        f"below the {MIN_CHECKPOINT_BYTES} byte minimum. "
        f"This usually means a partial write — do not trust this file."
    )
```

This catches:

- **Partial writes** (Ctrl-C during save — now fixed, see [Checkpointing](file:///home/sehaxe/busel-ai/site/src/content/docs/training/checkpointing.md))
- **Optimizer-only saves** (accidentally saved just the model dict)
- **Download corruption** (wget partial files)

If you hit this, **don't** try to load the file — it's not a valid checkpoint. Either:

- Resume training from an older checkpoint, OR
- Re-train (if the older checkpoints are also corrupt)

## The REPL

```bash
$ uv run python tools/inference.py --checkpoint checkpoints/ckpt_12000.pt --repl
ξ(◕ᴗ◕✿)ξ busel v5.2 — interactive REPL
Loaded checkpoint: shpak (11.0M params)
Type '/help' for commands, '/quit' to exit.

> Once upon a time
Once upon a time, the little stork flew over the village and saw...

> /temp 0.5
Temperature set to 0.5

> /top_k 100
Top-k set to 100

> The capital of France is
The capital of France is Paris, a city known for...

> /quit
ξ(≧◡≦)ξ Bye!
```

### REPL commands

| Command | Description |
|---|---|
| `/help` | List all commands |
| `/quit`, `/exit` | Exit the REPL |
| `/temp <float>` | Set temperature |
| `/top_k <int>` | Set top-k |
| `/top_p <float>` | Set top-p |
| `/rep_pen <float>` | Set repetition penalty |
| `/seed <int>` | Reset the seed |
| `/ctx` | Show current context length and remaining tokens |
| `/clear` | Clear the conversation history |
| `/save <path>` | Save the current conversation to a text file |
| `/load <path>` | Load a conversation from a text file |
| `/profile <step>` | Set the profile name (for display only) |

The conversation history is preserved across turns (within the context window). To start a fresh conversation, use `/clear`.

## Sampling parameters

### Temperature

```python
logits = logits / temperature
probs = softmax(logits)
```

- `T → 0`: greedy, deterministic, often repetitive
- `T = 0.8` (default): balanced
- `T = 1.2`: more diverse, more chaotic
- `T > 2.0`: usually garbage

For chat / instruct use `T = 0.7-0.9`. For creative writing use `T = 0.9-1.1`. For code use `T = 0.2-0.4`.

### Top-k

Keep only the top-k most probable tokens; sample from those.

- `k = 0`: off (sample from full distribution)
- `k = 50` (default): good for general use
- `k = 5-10`: focused, less diverse
- `k = 200+`: more diverse, more garbage

### Top-p (nucleus sampling)

Keep the smallest set of tokens whose cumulative probability is ≥ p.

- `p = 1.0`: off
- `p = 0.9` (default): good for general use
- `p = 0.5-0.7`: focused
- `p = 0.95-0.99`: more diverse

### Repetition penalty

Multiply the logits of already-seen tokens by `1/p` (penalty > 1) or `p` (penalty < 1).

- `1.0`: off
- `1.1` (default): light penalty
- `1.3+`: aggressive, may cause incoherence
- `< 1.0`: encourages repetition (rarely useful)

## The generation loop

```python
# tools/inference.py
def generate(model, input_ids, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
    """Standard autoregressive generation."""
    for _ in range(max_new_tokens):
        # Forward
        h_main, _ = model(input_ids)
        logits = h_main[:, -1, :]                     # last position
        
        # Apply repetition penalty
        for token_id in set(input_ids[0].tolist()):
            logits[:, token_id] /= repetition_penalty
        
        # Temperature
        logits = logits / temperature
        
        # Top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")
        
        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("inf")
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # EOS check (token 1 in our vocab, or byte 0x00 for some prompts)
        if next_token.item() in (1,):                  # adjust per your tokenizer
            break
    
    return input_ids
```

The loop uses only the **main head** (`lm_head`), not the 4 MTP heads. MTP-4 is a training signal, not an inference tool.

## KV cache for MLA

The MLA attention uses a 128-dim latent cache. Generation with `compile-mode=reduce-overhead` automatically captures CUDA graphs that cache the MLA compute.

For long generations (>1000 tokens), the GDN-2 linear attention uses O(1) state per head — no cache growth. The MLA blocks (every 4th) have O(S) cache.

## Batched generation

```python
# tools/inference.py
def generate_batch(model, prompts: list[str], max_new_tokens=200, ...):
    """Generate for multiple prompts in parallel."""
    input_ids = tokenizer.encode_batch(prompts)         # pad to same length
    # ... same loop as above, with batch dim
```

The CLI doesn't expose batched generation directly (the REPL is single-stream). For batched, use the Python API:

```python
from tools.inference import buselInferenceEngine

engine = buselInferenceEngine.from_checkpoint("checkpoints/ckpt_12000.pt")
outputs = engine.generate_batch(
    prompts=["Once upon a time", "In a galaxy far away", "The capital of France is"],
    max_new_tokens=100,
    temperature=0.8,
)
for prompt, output in zip(prompts, outputs):
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {output}\n")
```

## Performance

| Profile | Tokens/sec (greedy) | Tokens/sec (T=0.8) | First-token latency |
|---|---|---|---|
| shpak on RTX 5060 Ti | 580 | 540 | 18 ms |
| shpak on M2 Pro | 220 | 200 | 35 ms |
| shpak on M3 Max | 340 | 320 | 22 ms |
| zubr on RTX 4090 | 410 | 380 | 32 ms |
| chyzh on A100 | 850 | 820 | 55 ms |

The "first-token latency" is the prompt-processing time; subsequent tokens are bandwidth-bound.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselInferenceEngine` | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | The engine class |
| `generate()` | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | Autoregressive loop |
| `MIN_CHECKPOINT_BYTES` | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | The 10MB guard |
| REPL | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | Interactive mode |
| `generate_batch()` | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | Batched generation |
| `cli.py infer` | [cli.py](file:///home/sehaxe/busel-ai/cli.py) | The CLI subcommand |

## See also

- [Checkpointing](file:///home/sehaxe/busel-ai/site/src/content/docs/training/checkpointing.md) — checkpoint format
- [Architecture overview](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/overview.md) — what's in the model
- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md) — first training run

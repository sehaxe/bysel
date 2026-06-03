---
title: "Frequently asked questions"
description: "Common questions about busel — what it is, why 1.58-bit, why byte-level, how to extend it, licensing, and where to get help."
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

## General

### What is busel?

busel is a **sovereign 1.58-bit (1.58b) any-to-text LLM**. It implements the BitNet v2 architecture (ternary weights) with hybrid linear attention (GDN-2 / MLA), mAR (manifold-constrained attention residuals), MoE, byte-level patching (no BPE), and MTP-4 multi-token prediction. It's designed to train and infer on consumer hardware (16GB GPUs, Apple Silicon).

The name "busel" (бусел) is Belarusian for **stork**.

### Why 1.58-bit?

Three reasons:

1. **Memory efficiency** — ternary weights are 10× smaller than fp16. An 11M-parameter model is 12MB instead of 22MB.
2. **Inference speed** — ternary matmul is 2-5× faster than fp16 on consumer hardware (Apple's ANE, ARM NEON, AVX-512).
3. **Energy efficiency** — fewer bits switched = less power. Critical for on-device deployment.

The 1.58 number comes from `log₂(3)` — the information content of a ternary value.

### Why byte-level instead of BPE?

BPE / SentencePiece / tiktoken all require a tokenizer that's trained on your corpus. This means:

- **Tokenization drift** — adding a new corpus can change the vocab
- **Out-of-vocab** — rare characters become UNK tokens
- **English bias** — most BPE tokenizers are English-trained

byte-level avoids all of this:

- **No tokenizer** — the model reads raw bytes (UTF-8)
- **vocab_size = 259** — fixed forever (256 byte values + 3 multimodal specials)
- **Language-agnostic** — works on any UTF-8 text (Russian, Chinese, Arabic, emoji, etc.)
- **Multimodal** — images, audio, PDFs all become byte streams

The tradeoff: byte-level sequences are 4× longer than BPE for English. busel uses a stride-4 patcher to fold bytes back into longer tokens, getting most of the efficiency back.

### Why mAR (mHC + AttnRes) instead of standard residuals?

Standard additive residuals `x_{l+1} = x_l + f_l(x_l)` have three pathologies that 1-bit exposes:

1. **Unbounded magnitude** — `||x||` grows like `sqrt(L)`, breaking the quantizer
2. **Information bottleneck** — only 1 hop of cross-layer information
3. **Gradient confusion** — additive updates compose poorly

mAR (mHC + AttnRes) uses a **Birkhoff-projected mixing matrix** that combines the last 4 layer outputs via Sinkhorn-Knopp ×3. This:

1. **Bounds the magnitude** — doubly-stochastic matrices are bounded operators
2. **4-hop cross-layer info** — O(1) memory, O(L) effective depth
3. **Convex updates** — Birkhoff projection makes writes compose as convex combinations

See [mAR](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/mar.md) for the math.

### Can I add a new attention mechanism?

Yes. Use the [plugin registry](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/registry.md):

```python
from busel_registry import register

@register("attention", "mamba3")
class Mamba3Attention(nn.Module):
    ...
```

Add the file to `model/`, import it in `model/__init__.py`, set `attention_type: "mamba3"` in your config. No edits to `train.py` required.

### Can I use busel for [task X]?

busel is a **research framework**, not a production system. It works well for:

- ✅ Pre-training small models on custom corpora
- ✅ Byte-level research (1-bit, mAR, hybrid attention)
- ✅ On-device inference (16GB GPU, Apple Silicon)
- ✅ Educational use (read the code, learn 1-bit)

It does NOT work well for:

- ❌ Production chat / instruct models (use Llama, Mistral, Qwen)
- ❌ Image understanding (use LLaVA, Qwen-VL)
- ❌ Code completion (use CodeLlama, DeepSeek-Coder)
- ❌ Long-context (>8k tokens) — busel's max is 8192

## Technical

### How big is each profile?

| Profile | Parameters | Checkpoint | 16GB GPU? | Apple Silicon? |
|---|---|---|---|---|
| micro_test | 0.5M | 0.6 MB | ✅ | ✅ |
| quick_test | 2M | 2.3 MB | ✅ | ✅ |
| shpak | 11M | 12 MB | ✅ | ✅ (M1+) |
| zubr | 30M | 35 MB | ⚠️ (24GB needed) | ✅ (M2 Pro+) |
| chyzh | 165M | 180 MB | ❌ | ⚠️ (M2 Ultra+) |

### How long does training take?

For Shpak, end-to-end (12k steps, Chinchilla-optimal):

| Hardware | Time |
|---|---|
| RTX 5060 Ti (16GB) | ~1h 40m |
| RTX 4090 (24GB) | ~50m |
| M2 Pro | ~6h |
| M3 Max | ~4h |
| A100 40GB | ~25m |
| H100 80GB | ~15m |

For other profiles, scale linearly with the param count.

### How do I add a new profile?

Edit `configs/default.yaml`:

```yaml
my_profile:
  d_model: 640
  n_layers: 20
  # ... overrides only
```

Use it:

```bash
uv run train.py --profile my_profile
```

See [Config](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/config.md#how-to-add-a-new-profile).

### How do I use a custom dataset?

Drop files in `data_train/`. Supported: `.txt`, `.md`, `.jsonl`, `.parquet`, `.bin`, `.pdf`. See [Data formats](file:///home/sehaxe/busel-ai/site/src/content/docs/data/formats.md).

### How do I fine-tune a trained model?

```bash
# 1. Add your fine-tuning data to a new directory
mkdir -p data_finetune/
cp my_corpus.jsonl data_finetune/

# 2. Train with a smaller LR, starting from a checkpoint
uv run train.py --profile shpak \
                --data-dir data_finetune \
                --resume checkpoints/ckpt_12000.pt \
                --lr 0.0002 \
                --max-steps 2000
```

### How do I deploy the model?

busel v5.2 ships with `tools/inference.py` (CLI + REPL). The inference API server (FastAPI) was **removed** in v5.2 — the user wanted it as a separate repo. To deploy:

- **Local CLI** — `uv run python tools/inference.py --checkpoint X --repl`
- **Python API** — `from tools.inference import buselInferenceEngine`
- **Web service** — write your own FastAPI/Flask wrapper around `buselInferenceEngine.generate()`
- **Serverless** — bundle the inference script in a container, expose via HTTP

### Why is busel so small compared to Llama?

Llama 3.1 70B has 70 billion parameters. busel shpak has 11 million. That's 6000× smaller.

The reason: busel is **sovereign** (runs on your hardware) and **byte-level** (vocab = 259). Llama 3.1 is a frontier model trained on 15T tokens. Different use cases:

- **busel** — research, on-device, byte-level, sovereign
- **Llama 3.1** — production, cloud, BPE, frontier

You wouldn't use busel shpak to write a novel. You WOULD use busel shpak to:
- Pre-train on a domain corpus (medical, legal, code)
- Run inference on your laptop without internet
- Experiment with new architectures (1-bit, mAR, hybrid attention)

### What's the license?

**CC BY-NC-SA 4.0** with an additional restriction: **commercial use requires written permission from `sehaxe`**. See [LICENSE](file:///home/sehaxe/busel-ai/LICENSE) for the full text.

In short: you can use, modify, and redistribute busel for non-commercial purposes, with attribution. You cannot sell busel or use it to train commercial models without permission.

## Architecture deep-dives

### Why 3:1 GDN-2:MLA?

GDN-2 is O(1) per token (linear attention) but has limited expressiveness. MLA is O(S) per token (full attention) but is the most expressive.

The 3:1 ratio (75% GDN-2, 25% MLA) was found empirically to be the sweet spot for 1-bit models. Going to 1:1 (50/50) doubles the compute for ~2% perplexity improvement. Going to 7:1 (87.5/12.5) saves 30% compute for 5% perplexity regression.

### Why hybrid Muon + AdamW?

Pure Muon fails on 1D params (norms, biases) — orthogonalization doesn't make sense for scalars. Pure AdamW is 30% slower per step on dense transformers.

Hybrid routes 2D params to Muon (orthogonalized momentum) and 1D params to AdamW. The Muon update has unit norm, so the per-layer magnitude is uniform, which is exactly what 1-bit quantizers need.

### Why is the ctx warmup doubling, not linear?

Linear ctx growth (e.g., +128 per step) causes:
1. Continuous gradient distribution shift
2. Recompile thrashing (every step is a new shape)
3. The model never "settles" on a context length

Doubling (1024 → 2048 → 4096, 2000 steps each) gives the model time to reach a local equilibrium at each length. The cost is a single recompile per transition (30-60s), not constant.

## Operations

### Where do training logs go?

`checkpoints/busel.log.jsonl` — one JSON object per line. Use `jq` to query:

```bash
# Latest loss
jq -s 'max_by(.step) | {step, loss}' checkpoints/busel.log.jsonl

# All spike events
jq 'select(.event == "spike_detected")' checkpoints/busel.log.jsonl
```

See [Logging](file:///home/sehaxe/busel-ai/site/src/content/docs/reference/logging.md).

### How do I monitor training in real-time?

```bash
# Teto + JSON
tail -f checkpoints/busel.log.jsonl | jq -c '{ts, event, step, loss, lr}' | sed 's/^/ξ(◕ᴗ◕✿)ξ /'

# Plot it
uv run python tools/plotter.py --metric loss
```

### How do I run a single step and inspect?

```bash
# Profile a single step
uv run python cli.py profile --profile shpak --steps 1

# Or in Python
uv run python -c "
import torch
from busel_config import buselConfig
from model.backbone import buselModel

config = buselConfig.from_profile('shpak')
model = buselModel(config)
batch = torch.randint(0, 256, (1, 4096, 4), dtype=torch.uint8)
h_main, h_mtp = model(batch)
print('main logits:', h_main.shape)
print('mtp logits:', [h.shape for h in h_mtp])
"
```

### How do I run unit tests?

```bash
uv run python -m unittest discover tests
```

There are ~60 tests, runs in ~10 seconds. NEVER use pytest (this project uses unittest).

## Community

### Where can I get help?

- **GitHub Issues** — for bug reports and feature requests
- **GitHub Discussions** — for questions and ideas
- **The wiki** — you're reading it

### How do I report a bug?

Open a GitHub issue with:

1. The command you ran
2. The full error output
3. Your environment (`python --version`, `uv --version`, OS, GPU)
4. The contents of `checkpoints/busel.log.jsonl` (last 50 lines)

### How do I contribute?

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Open a PR

The tests must pass and the new code must follow the conventions in [AGENTS.md](file:///home/sehaxe/busel-ai/AGENTS.md).

### What's the project roadmap?

- **5.3** — Gradient checkpointing, audio byte encoding
- **5.4** — Multi-GPU (DDP)
- **6.0** — 1B parameter profile, RoPE scaling
- **6.x** — More attention mechanisms (Mamba-3, RWKV-7) as drop-in

### Where's the chat interface?

The user has explicitly said: **no chat interface in the main repo**. Chat UI / Telegram bot / web dashboard will be separate repos, communicating via the `busel.log.jsonl` event stream and the inference API.

## See also

- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md)
- [Troubleshooting](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/troubleshooting.md)
- [AGENTS.md](file:///home/sehaxe/busel-ai/AGENTS.md) — project knowledge base

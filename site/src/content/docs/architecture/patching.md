---
title: Byte-level patching (Gated FastBLT)
description: How 4 raw bytes become one model patch — and why we don't use BPE.
sidebar:
  order: 3
---

busel has no tokenizer. The model consumes raw UTF-8 bytes and the
`StridedFastBLTPatcher` compresses 4 bytes into one patch. This
page explains the patcher, the multimodal marker scheme, and why
the project refuses to add BPE.

## The tokenisation budget

A traditional BPE model has 32 000–200 000 tokens in its vocabulary.
For a 50 M-param model, **30–40 % of the parameters are in the
embedding matrix alone** — used once at the input, never again. The
BitNet team call this the *"embedding tax"*. Byte-level models
have a 259-token vocabulary and pay no such tax.

The cost is a 4× longer input sequence: an English sentence that
BPE encodes as 30 tokens becomes 120 bytes. The patcher compresses
those 120 bytes back into 30 patches, restoring the *effective*
context density while keeping the *vocabulary* tiny.

## The patcher

`StridedFastBLTPatcher` in `model/patching.py`:

```python
StridedFastBLTPatcher(stride=4, d_model=384)
```

The forward pass:

```text
  raw bytes  (B, T) uint8
      │
      │  nn.Embedding(259, d_byte)  ← learnable byte embedding
      ▼
  embedded  (B, T, d_byte)
      │
      │  Conv1d(d_byte, d_byte, kernel=5, stride=4, padding=3, groups=d_byte)
      │  ← causal-left padding:  F.pad(x, (3, 0))
      ▼
  conv out  (B, d_byte, T/4)
      │
      │  GLU gate:  sigmoid(gate_proj(x)) * up_proj(x)
      │  where gate_proj, up_proj : BitLinear_a4_8(d_byte, d_byte)
      ▼
  patches  (B, T/4, d_byte)
      │
      │  Linear projection: BitLinear_a4_8(d_byte, d_model)
      ▼
  patches  (B, T/4, d_model)
```

### Why a 1D conv with kernel=5?

`kernel=5, stride=4` means each patch sees itself plus 4 bytes of
left context (the receptive field). The `padding=3, groups=d_byte`
is a *causal* padding (left-only) implemented as `F.pad(x, (3, 0))`
before the conv. This gives every patch a 5-byte view of its
left context, with no leakage from the right (the next patch's
bytes).

The stride `4` is the byte-to-patch compression ratio. It is
**hard-coded**; changing it requires re-deriving the MTP-4 target
alignment in `train.py:build_targets`.

### Why the GLU gate?

A vanilla 1D conv would treat every byte equally. The gate
(sigmoid on a `BitLinear_a4_8(d_byte, d_byte)` projection, multiplied
element-wise with a parallel `BitLinear_a4_8(d_byte, d_byte)`)
learns to *suppress* bytes that don't carry information — typically
whitespace, markup, and repeated punctuation. This is busel's
response to the "spelling tax" of BPE: at the byte level we
*can* distinguish " " from "a" in 1.58 bits because the model
learns to ignore " ".

## The multimodal markers

Three "special" byte values mark non-text content in the byte
stream:

| Byte value | Meaning                                                       |
|-----------:|---------------------------------------------------------------|
| `256`      | Followed by 3 072 bytes = a 32×32×3 RGB image                 |
| `257`      | Followed by 8 bytes = a uint64 length prefix for a PDF page  |
| `258`      | Padding (zero-pad short chunks to `chunk_size`)               |

The image encoding is the work of `buselOmnivoreTextExtractor`
in `data/pipeline.py`: PIL loads the image, resizes to
`32×32×3 = 3 072` bytes, the byte `256` is prepended as a marker.
The model sees images as a special byte pattern in the same stream
as text.

PDFs go through Docling (if installed) → Markdown → bytes. They
are treated as ordinary text; the PDF page markers are mostly
informational at the data-loader level.

## Why no BPE

Three reasons:

1. **Parameter efficiency.** A 256-token embedding matrix for
   50 M params is 0.2 % of the model. A 32 000-token embedding
   would be 25 %.
2. **Robustness to noise.** BPE on out-of-vocabulary words falls
   back to character-level tokens, breaking consistency. Bytes
   never have an OOV.
3. **Multimodality for free.** Any file format is a byte stream;
   the model never has to know about JPEG vs PNG vs raw.

The trade-off is **sequence length** — a 4 096-byte input is only
1 024 patches. The patcher compresses back to 1 024 patches, but
you can never have more than 1 024 patches of context (≈ 1 024
* 4 = 4 096 bytes ≈ 1 024 English tokens). The MLA latent cache
helps (4 096 patches fits in ~98 MB), but busel is not the right
tool for million-token contexts.

## The hard constraint

`vocab_size` is exactly `259` everywhere it appears. The
embedding, the MTP head, the loss engine, the data loader — all
of them assume 256 byte values + 3 multimodal specials. Changing
this number is an anti-pattern; the model will silently misbehave
if you do.

## Where to look in the code

| Symbol                         | File                  | Role                          |
|--------------------------------|-----------------------|-------------------------------|
| `StridedFastBLTPatcher`        | `model/patching.py`   | The whole patcher             |
| `build_targets`                | `train.py`            | Aligns MTP targets to stride  |
| `buselOmnivoreTextExtractor`   | `data/pipeline.py`    | Image + PDF + JSON + parquet  |
| `RustByteStreamDataset`        | `data/pipeline.py`    | Mmap'd byte stream iterator   |

## See also

- [Architecture overview](/busel-ai/architecture/overview/) —
  where the patcher sits in the full pipeline.
- [Data → Multimodal encoding](/busel-ai/data/multimodal/) — how
  images and PDFs enter the byte stream.
- [Data → Pipeline](/busel-ai/data/pipeline/) — the loader side.

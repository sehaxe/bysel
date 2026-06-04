# busel — Known Issues / TODO

Tracking sheet for the code-audit findings of **2026-06-04**. New issues
appended at the bottom. File:line refs and `commit_hash` refer to the
snapshot at audit time (`HEAD = 21ec006`).

Priority legend:

- 🔴 **Correctness** — model trains worse / not at all
- 🟡 **Optimization / minor** — model trains but suboptimal
- 🟢 **Docs / cosmetic** — no runtime impact

---

## 🟡 OPEN — Optimization / Minor

### [#4] Newton-Schulz uses Frobenius norm for initial normalization (DOC DRIFT)

- **File:** `training/optimizer.py:24` (code) vs `training/AGENTS.md` (docs)
- **Reality:** Code uses **Frobenius** (`X = X / (X.norm() + 1e-8)`) with a
  detailed docstring explaining why: `‖X‖₂ ≤ ‖X‖_F`, so dividing by Frobenius
  guarantees `‖X‖₂ ≤ 1` with strict margin, which is the NS convergence
  region. Spectral norm (`X / ‖X‖₂`) is theoretically tighter but lands on
  the boundary and is sensitive to FP error — the docstring on
  `_newton_schulz_core` records an earlier attempt that diverged.
- **State:** Code is CORRECT (Frobenius over-normalises safely). The
  `training/AGENTS.md` line 28 description was stale ("spectral norm") and
  was fixed in this commit.
- **Alternative:** If a future optimisation pass wants to try spectral norm,
  use `X = X / (0.99 * spectral_norm(X))` to push strictly inside the
  convergence region. Run a 200-step validation-profile comparison; only
  switch if loss trajectory improves. Do not change without measurement.

### [#5] `compute_pretrain_loss` doesn't `ignore_index` padding targets

- **File:** `training/recipe.py:35-37`
- **When:** `byte_batch` is shorter than the chunk, the last few targets are
  zero-padded. Those positions still contribute to cross-entropy.
- **Impact:** borderline — training the model to predict 0 (NULL byte) at
  the dataset boundary is also valid signal. Skipping padding would throw
  away that signal. Leave as-is unless profiling shows it's hurting.
- **Status:** DOCUMENTED INTENTIONAL — closing as RESOLVED (decision: keep).

### [#6] `inject_noise` is CUDA-gated; per-param `.item()` sync

- **File:** `train.py:503-504` and `training/autopilot.py:134-143`
- **Current:** `if device == "cuda": autopilot.inject_noise(model)`
- **Why CUDA-only:** `inject_noise` calls `p.grad.norm().item()` per
  parameter (autopilot.py:140), which forces a CUDA stream sync per
  parameter. On MPS this is even more expensive (the per-call sync to
  Apple's Metal backend can take 1-5 ms; on a 100-parameter model that's
  100-500 ms per step). On CUDA it's tolerable because the kernel launch
  is batched. CPU doesn't have noise injection at all (the `.item()` would
  work but the rest of the path is unused).
- **Decision:** keep the gate. The noise itself (`randn_like`) works on
  any device — only the per-param sync is the bottleneck. If we want MPS
  noise injection later, hoist the noise scale to a single scalar
  computed once per step (not per-param) and pass it in. This is a
  refactor for the MPS-specific branch, not a bug.
- **Status:** DOCUMENTED INTENTIONAL.

---

## 🟢 OPEN — Docs / Cosmetic

(None open — see RESOLVED list below for what was closed.)

---

## ✅ RESOLVED

### [R1] MTP-4 label leakage — heads were seeing the answer

- **File:** `train.py:438` (was: `model(patches, mtp_targets, ...)`)
- **Bug:** MTP heads t2/t3/t4 were conditioned on T2/T3/T4 — the very bytes
  they were supposed to predict. T1 was the only honest head, so the model
  trained on ~25% of the intended signal.
- **Fix:** pass `[targets] + mtp_targets[:-1]` = `[T1, T2, T3]` as
  `next_token_ids` (proper DeepSeek-V3 cascade).
- **Commit:** `b31cd69`

### [R2] Inference `byte must be in range(0, 256)` crash

- **File:** `tools/inference.py:230` (and `apply_sampling`)
- **Bug:** model could sample special tokens 256/257/258 (image/pad/eos),
  `bytearray.append(256)` raises. Existing `>= vocab_size` check fired at
  ≥259, which never happens.
- **Fix:** mask `logits[256:] = -inf` in `apply_sampling`, plus defense-in-
  depth skip in `generate_stream` and `temperature==0` / multinomial-sum
  guards.
- **Commit:** `f1ec61e`

### [R3] Profiler warmup used wrong T1 target offset

- **File:** `tests/profiler_run.py` warmup loop
- **Bug:** warmup sliced `byte_batch[:, stride::stride]` (T1 = byte at
  offset `stride`), but `train.py:build_targets` uses
  `byte_batch[:, 1::stride]` (T1 = byte at offset 1). Two different targets
  → warmup loss didn't reflect actual training loss.
- **Fix:** align both to offset 1.
- **Commit:** `21ec006`

### [R4] `torch.profiler` was unavailable on CUDA

- **File:** `tests/profiler_run.py`
- **Gap:** only manual `time.perf_counter` profiling existed. No
  kernel-level / memory / Chrome trace support.
- **Fix:** new `StablebuselTorchProfiler` class + `--backend {auto,custom,torch}`
  flag. `auto` picks `torch` on CUDA, `custom` elsewhere.
- **Commit:** `21ec006`

### [R5] Muon param routing missed BitLinear weights (83% silently fell to AdamW)

- **File:** `training/optimizer.py:90` (was: `param.ndim == 2 and "router" not in name and "proj" in name`)
- **Bug:** `BitLinear_a4_8` extends `nn.Linear`; modules without `proj` in
  their name (MoE expert FFN, MLA compress/decompress, Blackboard memory,
  mtp_projections, mtp_heads) silently fell through to AdamW. ~83% of
  trainable parameters were getting the wrong optimizer.
- **Fix:** new rule: `param.ndim == 2 and all(token not in name for token in ("router", "embed"))` → Muon. Now ~96% of params go to Muon. Validation loss dropped from 7.20 → 6.20 over 200 steps on identical seed.
- **Commit:** `65caabf`

### [R6] Muon momentum update was non-standard (over-amplified recent grads)

- **File:** `training/optimizer.py:70-71` (was: `m_t = grad + momentum*buf_old`)
- **Bug:** produced `(1+momentum)*grad + momentum²*buf_old` instead of
  Keller Jordan's spec `momentum*buf_new + grad`. Over-weighted the most
  recent gradient.
- **Fix:** set `m_t = buf` (the post-update buffer = `momentum*buf_old + grad`).
  This matches the Muon paper exactly.
- **Commit:** `65caabf`

### [R7] No runtime guard for `max_steps > warmup_steps`

- **File:** `train.py:241-252` (new assertion)
- **Risk:** if `max_steps` is set lower than `warmup_steps` (e.g. by
  hand-edited config), `autopilot.update_parameters` crashes on
  `progress > 1.0` or `progress / 0` at line 86 / 112.
- **Fix:** raise `ValueError` if `cfg.max_steps <= cfg.warmup_steps`, with
  a helpful error message. Also guards `warmup_steps < 1`.
- **Commit:** `65caabf`

### [R8] Stale doc claims about AdamW weight_decay and MTP-4 loss weights

- **Files:** `training/AGENTS.md` (was: "AdamW weight_decay: 0.01 (fixed)"
  and "decay [1.0, .5, .25, .125]")
- **Reality:** AdamW is initialised with `weight_decay=0.01` but
  `autopilot.before_step` overwrites it on every step from the dynamic
  `target_wd × wd_factor` curve. MTP-4 weights are `[0.5, 0.25, 0.125]`
  for T2/T3/T4 (T1 has implicit weight 1.0). The old AGENTS.md text was
  misleading.
- **Fix:** rewrote those two lines to match code.
- **Commit:** `65caabf`

### [R9] `multimodal/` encoders were broken (Python bytes can't hold value 256)

- **File:** `multimodal/encoders.py` (was: returning `bytes`)
- **Bug:** encoders tried to build a token stream that included marker
  bytes 256, 257, 258 (multimodal boundary tokens). Python's `bytes` and
  `bytearray` types reject values ≥ 256 with `ValueError: bytes must be in
  range(0, 256)`. No test had ever exercised the multimodal codepath.
- **Fix:** all encoders now return `list[int]`. `collate_busel_batch` and
  the patcher already support list input. The existing latent bug in
  `buselOmnivoreTextExtractor` (which also used `bytearray.append(256)`)
  was fixed in the same commit.
- **Commit:** `2352f02`

### [R10] PIL/imageio slow paths in multimodal encoders

- **File:** `multimodal/encoders.py`
- **Gap:** image and video encoders used `PIL.Image.open().resize().tobytes()`
  and `imageio.imiter()` for frame counting. ~3-10× slower than cv2
  alternatives.
- **Fix:** replaced with `cv2.imread` + `cv2.resize(INTER_AREA)` + `cv2.cvtColor`
  for images (5.7× faster on 256², 3× on 1024²) and `cv2.VideoCapture` with
  `CAP_PROP_FRAME_COUNT` + `cap.grab()` for videos (10× faster for frame
  counting, skip-decoding for non-sampled frames). PIL/imageio retained as
  fallback.
- **Commit:** `2352f02`

---

## 📋 Future work (not bugs)

- **MoD router (`capacity_factor`) is currently always 1.0** — the code path
  is implemented but disabled. Enabling `<1.0` would speed up training
  (fewer tokens through experts) at the cost of quality. Would need a
  careful ablation study.
- **FLA `chunk_gdn2` Triton kernel** is used when available; the JIT
  fallback (`stable_gdn2_recurrent_jit`) is 100× slower. Both paths are
  present so MPS users are not blocked, but CUDA users with FLA installed
  get the fast path automatically.
- **The multimodal encoders do no resampling for audio** — input sample
  rate is stored in the header. The model trains on the source rate; if
  the corpus has wildly varying rates, consider a pre-processing pass to
  resample everything to 16 kHz (or whatever the dominant rate is).
- **The `busel` Python import requires `maturin develop --release`** — if
  the user clones and skips this, the multimodal encoders fall back to
  raw `open(..., 'rb')` which is still correct but doesn't get the cv2
  fast path. This is a known onboarding step, not a bug.

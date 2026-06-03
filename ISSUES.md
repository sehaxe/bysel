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

### [#1] Muon param routing may miss some BitLinear weights

- **File:** `training/optimizer.py:90`
- **Rule:** `param.ndim == 2 and "router" not in name and "proj" in name` → Muon
- **Risk:** `BitLinear_a4_8` extends `nn.Linear`; if a sub-module isn't named
  with `proj` in it, the weight silently falls through to AdamW. Need to
  check `model/attention.py` and `model/routing.py` to confirm that q/k/v/o
  projections and MoE expert weights all have `proj` in their path.
- **Fix:** if any are missed, broaden the rule to
  `param.ndim == 2 and "router" not in name and ("proj" in name or "weight" in name)`.
- **Effort:** 5 min (read 2 files + maybe 1 line change).

### [#2] AGENTS.md says "AdamW weight_decay: 0.01 (fixed)" but code makes it dynamic

- **Files:** `training/optimizer.py:105`, `training/autopilot.py:80-81`
- **Reality:** `autopilot.before_step` overwrites AdamW's `weight_decay` on
  every step from the dynamic `target_wd * wd_factor` curve.
- **Fix:** update `training/AGENTS.md` to reflect the dynamic behaviour.
- **Effort:** 2 min (docs only).

### [#3] Muon momentum update is non-standard

- **File:** `training/optimizer.py:70-71`
- **Current:** `buf = momentum*buf + grad;  m_t = grad + momentum*buf`
- **Keller Jordan reference:** `m_t = momentum*buf + grad`
- **Impact:** produces `(1+momentum)*grad + momentum²*buf_old` instead of
  the textbook form. Probably still converges (Muon is robust to momentum
  variations), but worth aligning with the spec.
- **Fix:** change line 71 to `m_t = buf.to(...)` (the updated buffer).
- **Effort:** 5 min.

### [#4] Newton-Schulz uses Frobenius norm for initial normalization

- **File:** `training/optimizer.py:19`
- **Current:** `X = X / (X.norm() + 1e-8)` (Frobenius, overestimates spectral)
- **Better:** spectral norm (`linalg.matrix_norm` with `ord=2` or
  power iteration)
- **Impact:** safe (over-normalization keeps spectral norm ≤ 1, which is
  the NS convergence region) but slightly slows convergence. Low priority.
- **Fix:** replace with `X = X / torch.linalg.matrix_norm(X, ord=2)` or a
  2-step power iteration. Verify on a quick Muon test.
- **Effort:** 15 min.

### [#5] `compute_pretrain_loss` doesn't `ignore_index` padding targets

- **File:** `training/recipe.py:35-37`
- **When:** `byte_batch` is shorter than the chunk, the last few targets are
  zero-padded. Those positions still contribute to cross-entropy.
- **Impact:** borderline — training the model to predict 0 (NULL byte) at
  the dataset boundary is also valid signal. Skipping padding would throw
  away that signal. Leave as-is unless profiling shows it's hurting.

### [#6] `inject_noise` is CUDA-only

- **File:** `train.py:485-486`
- **Current:** `if device == "cuda": autopilot.inject_noise(model)`
- **Suspected reason:** slow `randn_like` on MPS (also flagged in profiler
  recommendations). Unverified.
- **Fix:** either remove the CUDA gate (test on MPS first) or document why
  it's gated in `train.py` and the autopilot.
- **Effort:** 10 min (test + comment).

### [#7] No `max_steps > warmup_steps` runtime guard

- **File:** `train.py:217-234` (Chinchilla auto-planner)
- **Risk:** if `max_steps` is set lower than `warmup_steps` (e.g. by
  hand-edited config), `autopilot.update_parameters` crashes on
  `progress > 1.0` or `progress / 0` at line 86 / 112.
- **AGENTS.md says this is an anti-pattern** ("NEVER set `max_steps` <
  `warmup_steps` in any preset — produces NaN spikes") but there's no
  runtime check.
- **Fix:** add `assert cfg.max_steps > cfg.warmup_steps, "max_steps must be
  > warmup_steps"` right after computing the auto values.
- **Effort:** 3 min.

---

## 🟢 OPEN — Docs / Cosmetic

### [#8] MTP weights in AGENTS.md don't match code

- **Files:** `training/AGENTS.md` says "decay [1.0, .5, .25, .125]"; code at
  `training/recipe.py:45` uses `[0.5, 0.25, 0.125]`.
- **Reality:** code is correct (T1 has implicit weight 1.0; the 3 explicit
  weights are for T2, T3, T4). Doc is just confusingly worded.
- **Fix:** reword AGENTS.md to "T1 implicitly weighted 1.0; T2/T3/T4 use
  decaying weights [0.5, 0.25, 0.125]".
- **Effort:** 2 min.

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

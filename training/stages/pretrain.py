"""
🤖 busel PRETRAIN STAGE v1.0 — Base pretraining (next-byte CE on raw bytes)
The first stage of the pipeline. Trains a buselModel from scratch (or
resumes from a checkpoint) on byte-level data via MTP-4 + MoE + AutoPilot.

Extracted from train.py:main() so it can be invoked by the pipeline
orchestrator (tools/orchestrator.py:pipeline) in addition to the legacy
CLI mode. Behavior is preserved 1:1 with train.py.
"""
from __future__ import annotations

import os
import sys
import time
import signal
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from training.stages.base import BaseStage, StageState, register_stage, _apply_model_profile
from busel_logging import setup_logging, log_event


@dataclass
class buselPretrainConfig:
    """Subset of configs/default.yaml profile keys used by the pretrain stage.

    Mirrors the buselConfig class that used to live inside train.py. Each
    field corresponds to a key in the YAML profile; see configs/default.yaml
    for the canonical schema.
    """

    profile_name: str = "shpak"
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    expert_hidden: int = 256
    num_experts: int = 2
    top_k: int = 2
    vocab_size: int = 326
    n_hyper: int = 2
    data_path: str = "data_train"
    chunk_size: int = 256
    batch_size: int = 256
    weight_decay: float = 0.1
    max_steps: Any = "auto"
    warmup_steps: Any = "auto"
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    learning_rate_muon: float = 0.0006
    learning_rate_adamw: float = 0.00006
    use_ema: bool = False
    ema_decay: float = 0.999

    @classmethod
    def from_profile(cls, profile_dict: dict) -> "buselPretrainConfig":
        cfg = cls()
        m = profile_dict.get("model", {})
        d = profile_dict.get("data", {})
        t = profile_dict.get("training", {})
        _apply_model_profile(cfg, m)
        cfg.data_path = d.get("data_path", cfg.data_path)
        cfg.chunk_size = int(d.get("chunk_size", cfg.chunk_size))
        cfg.batch_size = int(d.get("batch_size", cfg.batch_size))
        cfg.weight_decay = float(t.get("weight_decay", cfg.weight_decay))
        cfg.max_steps = t.get("max_steps", cfg.max_steps)
        cfg.warmup_steps = t.get("warmup_steps", cfg.warmup_steps)
        cfg.min_lr_ratio = float(t.get("min_lr_ratio", cfg.min_lr_ratio))
        cfg.grad_accum_steps = int(t.get("grad_accum_steps", cfg.grad_accum_steps))
        cfg.learning_rate_muon = float(t.get("learning_rate_muon", cfg.learning_rate_muon))
        cfg.learning_rate_adamw = float(t.get("learning_rate_adamw", cfg.learning_rate_adamw))
        cfg.use_ema = bool(t.get("use_ema", cfg.use_ema))
        cfg.ema_decay = float(t.get("ema_decay", cfg.ema_decay))
        if cfg.d_model % cfg.n_hyper != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_hyper ({cfg.n_hyper})!"
            )
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})!"
            )
        return cfg


def _enforce_stability(seed: int = 42) -> None:
    """Set TF32, cuDNN benchmark, seed (mirrors train.py:enforce_stability)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_targets(byte_batch: torch.Tensor, input_length: int, stride: int = 4):
    """Compute MTP-4 targets (mirrors train.py:build_targets)."""
    targets = byte_batch[:, 1::stride][:, :input_length]
    if targets.shape[1] < input_length:
        pad = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad), value=0)
    mtp_targets = []
    for shift in (2, 3, 4):
        mt = byte_batch[:, shift::stride][:, :input_length]
        if mt.shape[1] < input_length:
            pad = input_length - mt.shape[1]
            mt = torch.nn.functional.pad(mt, (0, pad), value=0)
        mtp_targets.append(mt)
    return targets, mtp_targets


@register_stage("pretrain")
class buselPretrainStage:
    """Base pretraining stage.

    Lifecycle (per BaseStage Protocol):
        setup(cfg, profile_name, ...) → builds model, optimizer, dataloader
        run(state)                    → executes the training loop
        finalize(state)               → saves final checkpoint + log
    """

    name: str = "pretrain"

    def __init__(self) -> None:
        self.cfg: buselPretrainConfig | None = None
        self.profile_name: str = "shpak"
        self.device: str = "cpu"
        self.model: Any = None
        self.patcher: Any = None
        self.opt_engine: Any = None
        self.autopilot: Any = None
        self.loss_engine: Any = None
        self.dataloader: Any = None
        self.dataloader_iter: Any = None
        self.global_current_file_idx: int = 0
        self.global_current_byte_offset: int = 0
        self._compile_in_progress: dict = {"value": False}
        self._emergency_save_requested: dict = {"value": False}
        self.start_step: int = 0
        self.start_file_idx: int = 0
        self.start_byte_offset: int = 0
        self.no_compile: bool = False
        self.compile_mode: str = "default"
        self.no_checkpointing: bool = False
        self.ema: Any = None
        self._logger: Any = None

    def setup(
        self,
        profile: dict | str,
        profile_name: str = "shpak",
        *,
        resume: str | None = None,
        no_compile: bool = False,
        compile_mode: str = "default",
        no_checkpointing: bool = False,
        use_ema: bool | None = None,
        ema_decay: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize model + optimizer + dataloader for pretraining.

        Args:
            profile: Either a profile dict (from configs/default.yaml) or a
                profile NAME to look up. If a name is passed, configs/default.yaml
                is loaded and the matching profile is used.
            profile_name: Profile name to remember (used for logging + checkpoints).
            resume: Optional path to a checkpoint to resume from.
            no_compile: Disable torch.compile entirely.
            compile_mode: torch.compile mode (default|reduce-overhead|max-autotune).
            no_checkpointing: Disable gradient checkpointing.
        """
        stage_params = kwargs.pop("stage_params", None) or {}
        if use_ema is None:
            use_ema = stage_params.get("use_ema")
        if ema_decay is None:
            ema_decay = stage_params.get("ema_decay")
        if isinstance(profile, str):
            with open("configs/default.yaml", "r", encoding="utf-8") as f:
                full = yaml.safe_load(f)
            if profile not in full["profiles"]:
                raise ValueError(f"Profile {profile!r} not in configs/default.yaml")
            profile_dict = full["profiles"][profile]
            self.profile_name = profile
        else:
            profile_dict = profile

        self.cfg = buselPretrainConfig.from_profile(profile_dict)
        self.profile_name = profile_name
        self.no_compile = no_compile
        self.compile_mode = compile_mode
        self.no_checkpointing = no_checkpointing
        if use_ema is not None:
            self.cfg.use_ema = bool(use_ema)
        if ema_decay is not None:
            self.cfg.ema_decay = float(ema_decay)

        _enforce_stability()
        self._logger = setup_logging()
        log_event("training_start", profile=profile_name)

        self.device = _detect_device()

        if not os.path.exists(self.cfg.data_path):
            raise FileNotFoundError(f"Path {self.cfg.data_path!r} does not exist")

        from model.patching import StridedFastBLTPatcher
        from model.backbone import buselModel
        from training.optimizer import buselOptimizerEngine
        from training.autopilot import buselAutoPilot
        from training.recipe import buselLossEngine, validate_training_schedule

        self.patcher = StridedFastBLTPatcher(d_model=self.cfg.d_model).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        log_event(
            "model_initialized",
            profile=profile_name,
            device=self.device,
            total_params=total_params,
            model_size_mb=round(total_params * 2 / 1024**2, 2),
        )

        if self.cfg.max_steps == "auto" or self.cfg.max_steps is None:
            global_batch = self.cfg.batch_size * self.cfg.grad_accum_steps
            tokens_per_step = global_batch * (self.cfg.chunk_size // 4)
            chinchilla = 80 * total_params
            self.cfg.max_steps = math.ceil(chinchilla / tokens_per_step)
            log_event(
                "chinchilla_planned",
                target_tokens=chinchilla,
                planned_steps=self.cfg.max_steps,
                tokens_per_step=tokens_per_step,
                global_batch_size=global_batch,
            )
        else:
            self.cfg.max_steps = int(self.cfg.max_steps)

        if self.cfg.warmup_steps == "auto" or self.cfg.warmup_steps is None:
            self.cfg.warmup_steps = max(50, int(0.05 * self.cfg.max_steps))
        else:
            self.cfg.warmup_steps = int(self.cfg.warmup_steps)

        self.cfg.max_steps, self.cfg.warmup_steps = validate_training_schedule(
            self.cfg.max_steps, self.cfg.warmup_steps
        )

        if self.device == "cuda" and not self.no_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.device == "cuda" and not self.no_compile:
            self._compile_in_progress["value"] = True
            try:
                self.model = torch.compile(
                    self.model, fullgraph=False, dynamic=None, mode=self.compile_mode
                )
                self.patcher = torch.compile(
                    self.patcher, fullgraph=False, dynamic=None, mode=self.compile_mode
                )
            except Exception as e:
                err_str = str(e)
                if "CUDAGraphs" in err_str or "FakeTensor" in err_str or "overwritten" in err_str:
                    try:
                        self.model = torch.compile(self.model, fullgraph=False, dynamic=None)
                        self.patcher = torch.compile(self.patcher, fullgraph=False, dynamic=None)
                    except Exception:
                        pass
            finally:
                self._compile_in_progress["value"] = False

        self.opt_engine = buselOptimizerEngine(
            self.model,
            lr_muon=self.cfg.learning_rate_muon,
            lr_adamw=self.cfg.learning_rate_adamw,
        )
        self.autopilot = buselAutoPilot(
            self.opt_engine,
            max_lr_muon=self.cfg.learning_rate_muon,
            max_lr_adamw=self.cfg.learning_rate_adamw,
            target_wd=self.cfg.weight_decay,
            warmup_steps=self.cfg.warmup_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )
        self.loss_engine = buselLossEngine(self.cfg.vocab_size)

        if self.cfg.use_ema:
            from training.optimizer import EMA
            self.ema = EMA(self.model, decay=self.cfg.ema_decay)
            print(f"📈 EMA enabled: decay={self.cfg.ema_decay}")

        if resume and os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=self.device)
            from model.checkpoint import load_state_dict_safely
            load_state_dict_safely(self.model, checkpoint["model_state_dict"])
            load_state_dict_safely(self.patcher, checkpoint["patcher_state_dict"])
            if self.ema is not None and "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
            if checkpoint.get("step") != "emergency_backup":
                self.start_step = checkpoint["step"]
                self.start_file_idx = checkpoint.get("file_idx", 0)
                self.start_byte_offset = checkpoint.get("byte_offset", 0)

        from data.pipeline import get_busel_dataloader

        current_chunk_size = self.cfg.chunk_size // 4
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=current_chunk_size,
            batch_size=self.cfg.batch_size,
            start_file_idx=self.start_file_idx,
            start_byte_offset=self.start_byte_offset,
        )
        self.dataloader_iter = iter(self.dataloader)

        self.global_current_file_idx = self.start_file_idx
        self.global_current_byte_offset = self.start_byte_offset

        def _save_emergency_checkpoint(signum, frame):
            if self._compile_in_progress["value"] or self._emergency_save_requested["value"]:
                return
            self._emergency_save_requested["value"] = True
            try:
                log_event("emergency_save_requested", step=self.start_step, signal=signum)
            except Exception:
                pass

        signal.signal(signal.SIGINT, _save_emergency_checkpoint)
        signal.signal(signal.SIGTERM, _save_emergency_checkpoint)

    def run(self, state: StageState) -> StageState:
        """Execute the pretrain training loop for cfg.max_steps."""
        if self.cfg is None:
            raise RuntimeError("setup() must be called before run()")

        if self.device == "cuda" or self.device == "cpu":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
        autocast_enabled = self.device in ("cuda", "mps")

        use_cuda_stream = self.device == "cuda"
        prefetch_stream = torch.cuda.Stream() if use_cuda_stream else None
        current_batch: Any = None
        try:
            current_batch = next(self.dataloader_iter)
        except StopIteration:
            return state

        start_time = time.time()
        last_log_time = start_time
        last_log_tokens = 0
        current_chunk_size = self.cfg.chunk_size // 4
        current_batch_size = self.cfg.batch_size
        cumulative_processed_tokens = (
            self.start_step * current_batch_size * self.cfg.grad_accum_steps * current_chunk_size
        )

        for step_offset in range(self.cfg.max_steps):
            step = self.start_step + step_offset
            progress = float(step) / float(self.cfg.max_steps) if self.cfg.max_steps else 0.0

            new_chunk_size = current_chunk_size
            if progress < 0.15:
                new_chunk_size = self.cfg.chunk_size // 4
            elif progress < 0.35:
                new_chunk_size = self.cfg.chunk_size // 2
            else:
                new_chunk_size = self.cfg.chunk_size

            if new_chunk_size != current_chunk_size:
                new_batch_size = max(
                    1, (self.cfg.batch_size * (self.cfg.chunk_size // 4)) // new_chunk_size
                )
                current_chunk_size = new_chunk_size
                current_batch_size = new_batch_size

                from data.pipeline import get_busel_dataloader

                self.dataloader = get_busel_dataloader(
                    self.cfg.data_path,
                    chunk_size=current_chunk_size,
                    batch_size=current_batch_size,
                    start_file_idx=self.global_current_file_idx,
                    start_byte_offset=self.global_current_byte_offset,
                )
                self.dataloader_iter = iter(self.dataloader)
                try:
                    current_batch = next(self.dataloader_iter)
                except StopIteration:
                    break

            self.opt_engine.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0
            tokens_this_step = 0

            for _ in range(self.cfg.grad_accum_steps):
                if current_batch is None:
                    break
                byte_batch, last_file_idx, last_byte_offset = current_batch
                byte_batch = byte_batch.to(self.device, non_blocking=True)
                self.global_current_file_idx = last_file_idx
                self.global_current_byte_offset = last_byte_offset
                input_bytes = (
                    byte_batch[:, :-self.patcher.stride]
                    if byte_batch.shape[1] > self.patcher.stride
                    else byte_batch
                )

                with torch.autocast(
                    device_type=self.device, dtype=autocast_dtype, enabled=autocast_enabled
                ):
                    patches = self.patcher(input_bytes)
                    T_patches = patches.shape[1]
                    targets, mtp_targets = _build_targets(
                        byte_batch, T_patches, stride=self.patcher.stride
                    )
                    (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = self.model(
                        patches, [targets] + mtp_targets[:-1], progress=progress
                    )
                    loss = self.loss_engine.compute_pretrain_loss(
                        logits_t1, targets,
                        [logits_t2, logits_t3, logits_t4],
                        mtp_targets,
                    ) + aux_loss.float()

                loss = loss / self.cfg.grad_accum_steps
                loss.backward()
                accumulated_loss += loss.item() * self.cfg.grad_accum_steps
                accumulated_aux_loss += aux_loss.item()
                tokens_this_step = current_batch_size * current_chunk_size
                cumulative_processed_tokens += tokens_this_step

                next_batch = None
                if use_cuda_stream:
                    with torch.cuda.stream(prefetch_stream):
                        try:
                            next_batch = next(self.dataloader_iter)
                        except StopIteration:
                            next_batch = None
                else:
                    try:
                        next_batch = next(self.dataloader_iter)
                    except StopIteration:
                        next_batch = None
                if use_cuda_stream:
                    torch.cuda.current_stream().wait_stream(prefetch_stream)
                current_batch = next_batch

            if current_batch is None and accumulated_loss == 0.0:
                break

            dynamic_clip = self.autopilot.before_step(self.model, step, self.cfg.max_steps)
            if self.device == "cuda":
                self.autopilot.inject_noise(self.model)
            current_lr, _ = self.autopilot.update_parameters(step, accumulated_loss, self.cfg.max_steps)
            self.opt_engine.step()
            if self.ema is not None:
                self.ema.update(self.model)

            if self._emergency_save_requested["value"]:
                os.makedirs("checkpoints", exist_ok=True)
                try:
                    from model.checkpoint import strip_compile_prefix
                    torch.save(
                        {
                            "model_state_dict": strip_compile_prefix(self.model.state_dict()),
                            "patcher_state_dict": strip_compile_prefix(self.patcher.state_dict()),
                            "step": step,
                            "file_idx": self.global_current_file_idx,
                            "byte_offset": self.global_current_byte_offset,
                        },
                        "checkpoints/latest_crash_backup.pt",
                    )
                    log_event("emergency_checkpoint", step=step, path="checkpoints/latest_crash_backup.pt")
                except Exception as save_err:
                    print(f"❌ Emergency save failed: {type(save_err).__name__}: {save_err}")
                finally:
                    self._emergency_save_requested["value"] = False
                sys.exit(0)

            if step % 10 == 0:
                current_time = time.time()
                if step_offset == 0:
                    elapsed_interval = current_time - start_time
                    tokens_interval = tokens_this_step * self.cfg.grad_accum_steps
                else:
                    elapsed_interval = current_time - last_log_time
                    tokens_interval = cumulative_processed_tokens - last_log_tokens
                speed = tokens_interval / elapsed_interval if elapsed_interval > 0 else 0.0
                last_log_time = current_time
                last_log_tokens = cumulative_processed_tokens

                vram_mb = 0.0
                if self.device == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated() / 1024**2

                print(
                    f"Step {step:05d}/{self.cfg.max_steps:05d} | "
                    f"Total: {accumulated_loss:.2f} | "
                    f"Aux: {accumulated_aux_loss / max(1, self.cfg.grad_accum_steps):.2f} | "
                    f"LR: {current_lr:.5f} | "
                    f"Clip: {dynamic_clip:.2f} | "
                    f"Batch: {current_batch_size} | "
                    f"{speed:.0f} tokens/s"
                    + (f" | VRAM: {vram_mb:.0f}MB" if self.device == "cuda" else "")
                )

                os.makedirs("checkpoints", exist_ok=True)
                with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as log_f:
                    log_f.write(
                        json.dumps(
                            {
                                "step": step,
                                "loss": accumulated_loss,
                                "aux_loss": accumulated_loss / max(1, self.cfg.grad_accum_steps),
                                "lr": current_lr,
                                "speed": speed,
                                "vram": vram_mb,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                log_event(
                    "step_complete",
                    step=step,
                    loss=round(accumulated_loss, 4),
                    aux_loss=round(accumulated_aux_loss / max(1, self.cfg.grad_accum_steps), 4),
                    lr=round(current_lr, 7),
                    tokens_per_s=round(speed, 1),
                    vram_mb=round(vram_mb, 1),
                    batch=current_batch_size,
                    chunk=current_chunk_size,
                )

            if step % 100 == 0 and step > 0:
                self._save_scheduled_checkpoint(step, last_file_idx, last_byte_offset, accumulated_loss, current_lr)

            state.step = step
            state.best_loss = min(state.best_loss, accumulated_loss) if accumulated_loss > 0 else state.best_loss
            state.metrics = {
                "loss": accumulated_loss,
                "lr": current_lr,
                "tokens_per_s": speed if step % 10 == 0 else state.metrics.get("tokens_per_s", 0.0),
            }

        state.last_checkpoint_path = None
        return state

    def _save_scheduled_checkpoint(self, step, last_file_idx, last_byte_offset, accumulated_loss, current_lr) -> None:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/busel_{self.profile_name}_step_{step}.pt"
        temp_path = checkpoint_path + ".tmp"
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "patcher_state_dict": self.patcher.state_dict(),
            "step": step,
            "file_idx": last_file_idx,
            "byte_offset": last_byte_offset,
            "loss": accumulated_loss,
            "lr_muon": current_lr,
            "profile": self.profile_name,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        try:
            torch.save(ckpt, temp_path)
            file_size = os.path.getsize(temp_path)
            if file_size < 2_000_000:
                os.remove(temp_path)
                log_event("checkpoint_rejected", step=step, file_size=file_size, reason="too_small")
            else:
                os.rename(temp_path, checkpoint_path)
                print(f"💾 Scheduled checkpoint saved: {checkpoint_path} ({file_size / 1024 / 1024:.1f} MB)")
                log_event(
                    "checkpoint_saved",
                    step=step,
                    path=checkpoint_path,
                    file_size_mb=round(file_size / 1024 / 1024, 2),
                    loss=round(accumulated_loss, 4),
                    lr=round(current_lr, 7),
                )
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            import logging as _logging
            log_event("checkpoint_failed", step=step, error=str(e), level=_logging.WARNING)

    def finalize(self, state: StageState) -> StageState:
        """Save the final checkpoint + emit stage_complete event."""
        if self.cfg is None or self.model is None:
            return state

        os.makedirs("checkpoints", exist_ok=True)
        final_path = f"checkpoints/busel_{self.profile_name}_FINAL.pt"
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "patcher_state_dict": self.patcher.state_dict(),
            "step": state.step,
            "file_idx": self.global_current_file_idx,
            "byte_offset": self.global_current_byte_offset,
            "profile": self.profile_name,
            "config": self.cfg.__dict__,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        try:
            torch.save(ckpt, final_path)
            print(f"💾 Final checkpoint: {final_path}")
            log_event(
                "stage_complete",
                stage=self.name,
                profile=self.profile_name,
                total_steps=state.step,
                final_path=final_path,
            )
            state.last_checkpoint_path = final_path
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

        return state

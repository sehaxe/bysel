"""
🤖 busel SFT STAGE v1.0 — Supervised fine-tuning on chat-formatted data
The second stage of the pipeline. Resumes from a pretrain checkpoint and
fine-tunes on multi-turn chat JSONL with **assistant-only loss masking**.

Architecture: identical to `buselPretrainStage` but:
- Uses `data.sft.get_sft_dataloader` (yields (bytes, mask) batches)
- Uses `buselLossEngine.compute_sft_loss` (masked CE on t+1 head only)
- Default LR is 0.3x the pretrain LR (standard SFT recipe)
- No MTP heads (single t+1 prediction per token)
- No mAR `progress` propagation (not needed for SFT)
"""
from __future__ import annotations

import os
import time
import json
import signal
import math
from dataclasses import dataclass
from typing import Any

import torch
import yaml

from training.stages.base import BaseStage, StageState, register_stage
from busel_logging import setup_logging, log_event


@dataclass
class buselSFTConfig:
    """Subset of configs/default.yaml profile keys used by the SFT stage.

    Mirrors buselPretrainConfig but with SFT-tuned defaults (lower LR,
    no MTP, no mAR progress). Reuses the same d_model / n_layers / vocab
    keys from the profile so the loaded checkpoint's shape matches.
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
    data_glob: str = "data_train/sft/**/*.jsonl"
    chunk_size: int = 256
    batch_size: int = 256
    weight_decay: float = 0.1
    max_steps: Any = "auto"
    warmup_steps: Any = "auto"
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    learning_rate_muon: float = 0.00018
    learning_rate_adamw: float = 0.000018
    sft_lr_scale: float = 0.3
    data_paths: list[str] | None = None

    @classmethod
    def from_profile(cls, profile_dict: dict, stage_params: dict | None = None) -> "buselSFTConfig":
        cfg = cls()
        sp = dict(stage_params or {})
        m = profile_dict.get("model", {})
        d = profile_dict.get("data", {})
        t = profile_dict.get("training", {})
        cfg.d_model = int(m.get("d_model", cfg.d_model))
        cfg.n_layers = int(m.get("n_layers", cfg.n_layers))
        cfg.n_heads = int(m.get("n_heads", cfg.n_heads))
        cfg.expert_hidden = int(m.get("expert_hidden", cfg.expert_hidden))
        cfg.num_experts = int(m.get("num_experts", cfg.num_experts))
        cfg.top_k = int(m.get("top_k", cfg.top_k))
        cfg.vocab_size = int(m.get("vocab_size", cfg.vocab_size))
        cfg.n_hyper = int(m.get("n_hyper", cfg.n_hyper))
        cfg.data_glob = d.get("sft_data_glob", d.get("data_glob", cfg.data_glob))
        cfg.chunk_size = int(d.get("chunk_size", cfg.chunk_size))
        cfg.batch_size = int(d.get("sft_batch_size", d.get("batch_size", cfg.batch_size)))
        cfg.weight_decay = float(t.get("weight_decay", cfg.weight_decay))
        cfg.max_steps = sp.get("max_steps", t.get("sft_max_steps", t.get("max_steps", cfg.max_steps)))
        cfg.warmup_steps = sp.get("warmup_steps", t.get("sft_warmup_steps", t.get("warmup_steps", cfg.warmup_steps)))
        cfg.min_lr_ratio = float(t.get("min_lr_ratio", cfg.min_lr_ratio))
        cfg.grad_accum_steps = int(t.get("grad_accum_steps", cfg.grad_accum_steps))
        base_muon = float(t.get("learning_rate_muon", cfg.learning_rate_muon / cfg.sft_lr_scale))
        base_adamw = float(t.get("learning_rate_adamw", cfg.learning_rate_adamw / cfg.sft_lr_scale))
        cfg.sft_lr_scale = float(sp.get("sft_lr_scale", cfg.sft_lr_scale))
        cfg.learning_rate_muon = base_muon * cfg.sft_lr_scale
        cfg.learning_rate_adamw = base_adamw * cfg.sft_lr_scale
        cfg.data_paths = sp.get("data_paths", None)
        if cfg.d_model % cfg.n_hyper != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_hyper ({cfg.n_hyper})")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
        return cfg


def _strip_compile_prefix(sd: dict) -> dict:
    """Strip torch.compile prefixes for portable checkpoint loading."""
    if not sd:
        return sd
    out: dict = {}
    for k, v in sd.items():
        new_k = k
        for prefix in ("_orig_mod.", "compiled_model.", "_dynamo."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        out[new_k] = v
    return out


def _enforce_stability(seed: int = 42) -> None:
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


@register_stage("sft")
class buselSFTStage:
    """Supervised fine-tuning stage. Resumes from a pretrain checkpoint."""

    name: str = "sft"

    def __init__(self) -> None:
        self.cfg: buselSFTConfig | None = None
        self.profile_name: str = "shpak"
        self.device: str = "cpu"
        self.model: Any = None
        self.patcher: Any = None
        self.opt_engine: Any = None
        self.autopilot: Any = None
        self.loss_engine: Any = None
        self.dataloader: Any = None
        self.dataloader_iter: Any = None
        self.start_step: int = 0
        self.no_compile: bool = False
        self.compile_mode: str = "default"
        self.no_checkpointing: bool = False
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
        stage_params: dict | None = None,
    ) -> None:
        if isinstance(profile, str):
            with open("configs/default.yaml", "r", encoding="utf-8") as f:
                full = yaml.safe_load(f)
            if profile not in full["profiles"]:
                raise ValueError(f"Profile {profile!r} not in configs/default.yaml")
            profile_dict = full["profiles"][profile]
        else:
            profile_dict = profile

        self.cfg = buselSFTConfig.from_profile(profile_dict, stage_params)
        self.profile_name = profile_name
        self.no_compile = no_compile
        self.compile_mode = compile_mode
        self.no_checkpointing = no_checkpointing

        _enforce_stability()
        self._logger = setup_logging()
        log_event("sft_start", profile=profile_name)

        self.device = _detect_device()

        from model.patching import StridedFastBLTPatcher
        from model.backbone import buselModel
        from training.optimizer import buselOptimizerEngine
        from training.autopilot import buselAutoPilot
        from training.recipe import buselLossEngine, validate_training_schedule

        self.patcher = StridedFastBLTPatcher(d_model=self.cfg.d_model).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        log_event(
            "sft_model_initialized",
            profile=profile_name,
            device=self.device,
            total_params=total_params,
        )

        if self.cfg.max_steps == "auto" or self.cfg.max_steps is None:
            estimated_examples = self._estimate_examples_count()
            steps_per_epoch = max(1, estimated_examples // max(1, self.cfg.batch_size))
            self.cfg.max_steps = max(200, int(steps_per_epoch * 3))
            log_event(
                "sft_steps_planned",
                planned_steps=self.cfg.max_steps,
                estimated_examples=estimated_examples,
            )
        else:
            self.cfg.max_steps = int(self.cfg.max_steps)

        if self.cfg.warmup_steps == "auto" or self.cfg.warmup_steps is None:
            self.cfg.warmup_steps = max(20, int(0.05 * self.cfg.max_steps))
        else:
            self.cfg.warmup_steps = int(self.cfg.warmup_steps)

        self.cfg.max_steps, self.cfg.warmup_steps = validate_training_schedule(
            self.cfg.max_steps, self.cfg.warmup_steps
        )

        if self.device == "cuda" and not self.no_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.device == "cuda" and not self.no_compile:
            try:
                self.model = torch.compile(self.model, fullgraph=False, dynamic=None, mode=self.compile_mode)
                self.patcher = torch.compile(self.patcher, fullgraph=False, dynamic=None, mode=self.compile_mode)
            except Exception:
                pass

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

        if resume and os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))
            self.patcher.load_state_dict(_strip_compile_prefix(checkpoint["patcher_state_dict"]))
            self.start_step = checkpoint.get("step", 0) if checkpoint.get("step") != "emergency_backup" else 0
            print(f"📥 Resumed SFT from {resume} at step {self.start_step}")
            log_event("sft_resumed", resume=resume, start_step=self.start_step)

        from data.sft import get_sft_dataloader

        data_paths = self.cfg.data_paths or self.cfg.data_glob
        current_chunk_size = self.cfg.chunk_size // 4
        self.dataloader = get_sft_dataloader(
            data_paths,
            chunk_size=current_chunk_size,
            batch_size=self.cfg.batch_size,
        )
        self.dataloader_iter = iter(self.dataloader)

    def _estimate_examples_count(self) -> int:
        """Count lines in the SFT JSONL files (cheap, no parse)."""
        import glob
        if self.cfg.data_paths:
            files = []
            for p in self.cfg.data_paths:
                if os.path.isdir(p):
                    files.extend(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
                else:
                    files.extend(glob.glob(p, recursive=True))
        else:
            files = glob.glob(self.cfg.data_glob, recursive=True)
        files = [f for f in files if f.endswith(".jsonl")]
        total = 0
        for f in files:
            try:
                with open(f, "rb") as fh:
                    total += sum(1 for _ in fh)
            except OSError:
                pass
        return total if total > 0 else 1000

    def run(self, state: StageState) -> StageState:
        if self.cfg is None:
            raise RuntimeError("setup() must be called before run()")

        if self.device == "cuda" or self.device == "cpu":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
        autocast_enabled = self.device in ("cuda", "mps")

        current_chunk_size = self.cfg.chunk_size // 4
        current_batch_size = self.cfg.batch_size

        for step_offset in range(self.cfg.max_steps):
            step = self.start_step + step_offset
            progress = float(step) / float(self.cfg.max_steps) if self.cfg.max_steps else 0.0

            self.opt_engine.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            tokens_this_step = 0

            for _ in range(self.cfg.grad_accum_steps):
                try:
                    byte_batch, mask_batch = next(self.dataloader_iter)
                except StopIteration:
                    self.dataloader_iter = iter(self.dataloader)
                    byte_batch, mask_batch = next(self.dataloader_iter)

                byte_batch = byte_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                input_bytes = (
                    byte_batch[:, :-self.patcher.stride]
                    if byte_batch.shape[1] > self.patcher.stride
                    else byte_batch
                )

                with torch.autocast(
                    device_type=self.device, dtype=autocast_dtype, enabled=autocast_enabled
                ):
                    patches = self.patcher(input_bytes)
                    (logits_t1, _, _, _), aux_loss = self.model(patches, None, progress=progress)
                    shift = self.patcher.stride
                    targets = byte_batch[:, shift:shift + patches.shape[1]]
                    if targets.shape[1] < patches.shape[1]:
                        targets = torch.nn.functional.pad(
                            targets,
                            (0, patches.shape[1] - targets.shape[1]),
                            value=0,
                        )
                    mask_for_loss = mask_batch[:, shift:shift + patches.shape[1]]
                    if mask_for_loss.shape[1] < patches.shape[1]:
                        mask_for_loss = torch.nn.functional.pad(
                            mask_for_loss,
                            (0, patches.shape[1] - mask_for_loss.shape[1]),
                            value=0,
                        )
                    loss = self.loss_engine.compute_sft_loss(
                        logits_t1, targets, mask_for_loss
                    ) + aux_loss.float()

                loss = loss / self.cfg.grad_accum_steps
                loss.backward()
                accumulated_loss += loss.item() * self.cfg.grad_accum_steps
                tokens_this_step = current_batch_size * current_chunk_size

            dynamic_clip = self.autopilot.before_step(self.model, step, self.cfg.max_steps)
            current_lr, _ = self.autopilot.update_parameters(step, accumulated_loss, self.cfg.max_steps)
            self.opt_engine.step()

            if step % 10 == 0:
                vram_mb = 0.0
                if self.device == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                print(
                    f"[SFT] Step {step:05d}/{self.cfg.max_steps:05d} | "
                    f"Loss: {accumulated_loss:.3f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Clip: {dynamic_clip:.2f}"
                    + (f" | VRAM: {vram_mb:.0f}MB" if self.device == "cuda" else "")
                )
                with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "sft", "step": step, "loss": accumulated_loss,
                        "lr": current_lr, "vram": vram_mb,
                    }, ensure_ascii=False) + "\n")
                log_event("sft_step_complete", step=step, loss=round(accumulated_loss, 4),
                          lr=round(current_lr, 7), vram_mb=round(vram_mb, 1))

            state.step = step
            state.best_loss = min(state.best_loss, accumulated_loss) if accumulated_loss > 0 else state.best_loss
            state.metrics = {"loss": accumulated_loss, "lr": current_lr}

        return state

    def finalize(self, state: StageState) -> StageState:
        if self.cfg is None or self.model is None:
            return state
        os.makedirs("checkpoints", exist_ok=True)
        final_path = f"checkpoints/busel_{self.profile_name}_SFT_FINAL.pt"
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "patcher_state_dict": self.patcher.state_dict(),
                "step": state.step,
                "profile": self.profile_name,
                "config": self.cfg.__dict__,
                "stage": "sft",
            }, final_path)
            print(f"💾 SFT final checkpoint: {final_path}")
            log_event("stage_complete", stage=self.name, profile=self.profile_name,
                      total_steps=state.step, final_path=final_path)
            state.last_checkpoint_path = final_path
        except Exception as e:
            print(f"❌ Failed to save SFT checkpoint: {e}")
        return state

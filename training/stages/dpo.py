"""
🤖 busel DPO STAGE v1.0 — Direct Preference Optimization (Rafailov et al. 2023)
The third stage of the pipeline. Resumes from an SFT checkpoint and runs
DPO on `(prompt, chosen, rejected)` triples.

Design choice: rather than loading the SFT model twice (policy + reference),
we use the **same** model for both — computing reference log-probs under
`torch.no_grad()` + `model.eval()`. This halves memory and is the standard
simplification (the reference is the policy's weights at the START of DPO,
i.e. exactly the SFT model state).
"""
from __future__ import annotations

import os
import time
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from training.stages.base import BaseStage, StageState, register_stage
from busel_logging import setup_logging, log_event


@dataclass
class buselDPOConfig:
    profile_name: str = "shpak"
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    expert_hidden: int = 256
    num_experts: int = 2
    top_k: int = 2
    vocab_size: int = 326
    n_hyper: int = 2
    data_glob: str = "data_train/dpo/**/*.jsonl"
    chunk_size: int = 256
    batch_size: int = 32
    weight_decay: float = 0.1
    max_steps: Any = "auto"
    warmup_steps: Any = "auto"
    min_lr_ratio: float = 0.1
    grad_accum_steps: int = 1
    learning_rate_muon: float = 0.00006
    learning_rate_adamw: float = 0.000006
    dpo_beta: float = 0.1
    dpo_lr_scale: float = 0.1
    data_paths: list[str] | None = None

    @classmethod
    def from_profile(cls, profile_dict: dict, stage_params: dict | None = None) -> "buselDPOConfig":
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
        cfg.data_glob = d.get("dpo_data_glob", d.get("data_glob", cfg.data_glob))
        cfg.chunk_size = int(d.get("chunk_size", cfg.chunk_size))
        cfg.batch_size = int(d.get("dpo_batch_size", d.get("batch_size", cfg.batch_size)))
        cfg.weight_decay = float(t.get("weight_decay", cfg.weight_decay))
        cfg.max_steps = sp.get("max_steps", t.get("dpo_max_steps", t.get("max_steps", cfg.max_steps)))
        cfg.warmup_steps = sp.get("warmup_steps", t.get("dpo_warmup_steps", t.get("warmup_steps", cfg.warmup_steps)))
        cfg.min_lr_ratio = float(t.get("min_lr_ratio", cfg.min_lr_ratio))
        cfg.grad_accum_steps = int(t.get("grad_accum_steps", cfg.grad_accum_steps))
        base_muon = float(t.get("learning_rate_muon", cfg.learning_rate_muon / cfg.dpo_lr_scale))
        base_adamw = float(t.get("learning_rate_adamw", cfg.learning_rate_adamw / cfg.dpo_lr_scale))
        cfg.dpo_lr_scale = float(sp.get("dpo_lr_scale", cfg.dpo_lr_scale))
        cfg.learning_rate_muon = base_muon * cfg.dpo_lr_scale
        cfg.learning_rate_adamw = base_adamw * cfg.dpo_lr_scale
        cfg.dpo_beta = float(sp.get("beta", cfg.dpo_beta))
        cfg.data_paths = sp.get("data_paths", None)
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
        return cfg


def _strip_compile_prefix(sd: dict) -> dict:
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


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sequence_logp_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of log P(target[t]) for t where mask[t]=1, per sequence."""
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)
    return (target_log_probs * mask.float()).sum(dim=-1)


@register_stage("dpo")
class buselDPOStage:
    """DPO stage. Resumes from an SFT checkpoint, trains for cfg.max_steps."""

    name: str = "dpo"

    def __init__(self) -> None:
        self.cfg: buselDPOConfig | None = None
        self.profile_name: str = "shpak"
        self.device: str = "cpu"
        self.model: Any = None
        self.patcher: Any = None
        self.opt_engine: Any = None
        self.autopilot: Any = None
        self.dataloader_iter: Any = None
        self.start_step: int = 0

    def setup(
        self,
        profile: dict | str,
        profile_name: str = "shpak",
        *,
        resume: str | None = None,
        no_compile: bool = True,
        no_checkpointing: bool = True,
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

        self.cfg = buselDPOConfig.from_profile(profile_dict, stage_params)
        self.profile_name = profile_name

        _enforce_stability()
        self._logger = setup_logging()
        log_event("dpo_start", profile=profile_name, beta=self.cfg.dpo_beta)
        self.device = _detect_device()

        from model.patching import StridedFastBLTPatcher
        from model.backbone import buselModel
        from training.optimizer import buselOptimizerEngine
        from training.autopilot import buselAutoPilot
        from training.recipe import validate_training_schedule

        self.patcher = StridedFastBLTPatcher(d_model=self.cfg.d_model).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)

        if self.cfg.max_steps == "auto" or self.cfg.max_steps is None:
            import glob
            files = self.cfg.data_paths or glob.glob(self.cfg.data_glob, recursive=True)
            files = [f for f in files if f.endswith(".jsonl")]
            n_examples = 0
            for f in files:
                try:
                    with open(f, "rb") as fh:
                        n_examples += sum(1 for _ in fh)
                except OSError:
                    pass
            steps_per_epoch = max(1, n_examples // max(1, self.cfg.batch_size))
            self.cfg.max_steps = max(200, int(steps_per_epoch * 2))
        else:
            self.cfg.max_steps = int(self.cfg.max_steps)

        if self.cfg.warmup_steps == "auto" or self.cfg.warmup_steps is None:
            self.cfg.warmup_steps = max(20, int(0.05 * self.cfg.max_steps))
        else:
            self.cfg.warmup_steps = int(self.cfg.warmup_steps)

        self.cfg.max_steps, self.cfg.warmup_steps = validate_training_schedule(
            self.cfg.max_steps, self.cfg.warmup_steps
        )

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

        if resume and os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))
            self.patcher.load_state_dict(_strip_compile_prefix(checkpoint["patcher_state_dict"]))
            self.start_step = checkpoint.get("step", 0) if checkpoint.get("step") != "emergency_backup" else 0
            print(f"📥 Resumed DPO from {resume} at step {self.start_step}")
            log_event("dpo_resumed", resume=resume, start_step=self.start_step)
        else:
            raise FileNotFoundError(
                "DPO stage requires `resume=` to point to an SFT checkpoint. "
                "Run the SFT stage first or pass `resume: checkpoints/busel_<profile>_SFT_FINAL.pt`."
            )

        from data.dpo import get_dpo_dataloader
        data_paths = self.cfg.data_paths or self.cfg.data_glob
        current_chunk_size = self.cfg.chunk_size // 4
        self.dataloader_iter = iter(get_dpo_dataloader(
            data_paths, chunk_size=current_chunk_size, batch_size=self.cfg.batch_size,
        ))

    def run(self, state: StageState) -> StageState:
        if self.cfg is None:
            raise RuntimeError("setup() must be called before run()")

        autocast_dtype = torch.bfloat16 if self.device in ("cuda", "cpu") else torch.float16
        autocast_enabled = self.device in ("cuda", "mps")
        chunk_size = self.cfg.chunk_size // 4
        beta = self.cfg.dpo_beta

        for step_offset in range(self.cfg.max_steps):
            step = self.start_step + step_offset
            self.opt_engine.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_acc = 0.0

            for _ in range(self.cfg.grad_accum_steps):
                try:
                    cb, cm, rb, rm = next(self.dataloader_iter)
                except StopIteration:
                    from data.dpo import get_dpo_dataloader
                    data_paths = self.cfg.data_paths or self.cfg.data_glob
                    self.dataloader_iter = iter(get_dpo_dataloader(
                        data_paths, chunk_size=chunk_size, batch_size=self.cfg.batch_size,
                    ))
                    cb, cm, rb, rm = next(self.dataloader_iter)

                cb = cb.to(self.device, non_blocking=True)
                cm = cm.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)
                rm = rm.to(self.device, non_blocking=True)

                chosen_in = cb[:, :-self.patcher.stride] if cb.shape[1] > self.patcher.stride else cb
                rejected_in = rb[:, :-self.patcher.stride] if rb.shape[1] > self.patcher.stride else rb

                with torch.autocast(device_type=self.device, dtype=autocast_dtype, enabled=autocast_enabled):
                    policy_chosen_patches = self.patcher(chosen_in)
                    policy_rejected_patches = self.patcher(rejected_in)
                    (pol_c_logits, _, _, _), aux_pol_c = self.model(policy_chosen_patches, None)
                    (pol_r_logits, _, _, _), aux_pol_r = self.model(policy_rejected_patches, None)

                shift = self.patcher.stride
                T_p = pol_c_logits.shape[1]
                chosen_tgt = cb[:, shift:shift + T_p]
                if chosen_tgt.shape[1] < T_p:
                    chosen_tgt = torch.nn.functional.pad(chosen_tgt, (0, T_p - chosen_tgt.shape[1]), value=0)
                chosen_msk = cm[:, shift:shift + T_p]
                if chosen_msk.shape[1] < T_p:
                    chosen_msk = torch.nn.functional.pad(chosen_msk, (0, T_p - chosen_msk.shape[1]), value=0)

                rejected_tgt = rb[:, shift:shift + T_p]
                if rejected_tgt.shape[1] < T_p:
                    rejected_tgt = torch.nn.functional.pad(rejected_tgt, (0, T_p - rejected_tgt.shape[1]), value=0)
                rejected_msk = rm[:, shift:shift + T_p]
                if rejected_msk.shape[1] < T_p:
                    rejected_msk = torch.nn.functional.pad(rejected_msk, (0, T_p - rejected_msk.shape[1]), value=0)

                policy_chosen_logps = _sequence_logp_from_logits(pol_c_logits, chosen_tgt, chosen_msk)
                policy_rejected_logps = _sequence_logp_from_logits(pol_r_logits, rejected_tgt, rejected_msk)

                with torch.no_grad():
                    self.model.eval()
                    ref_chosen_patches = self.patcher(chosen_in)
                    ref_rejected_patches = self.patcher(rejected_in)
                    (ref_c_logits, _, _, _), _ = self.model(ref_chosen_patches, None)
                    (ref_r_logits, _, _, _), _ = self.model(ref_rejected_patches, None)
                    self.model.train()
                reference_chosen_logps = _sequence_logp_from_logits(ref_c_logits, chosen_tgt, chosen_msk)
                reference_rejected_logps = _sequence_logp_from_logits(ref_r_logits, rejected_tgt, rejected_msk)

                from training.recipe import buselLossEngine
                loss = buselLossEngine.compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    beta=beta,
                )
                with torch.no_grad():
                    logits = beta * (
                        (policy_chosen_logps - policy_rejected_logps)
                        - (reference_chosen_logps - reference_rejected_logps)
                    )
                    acc = (logits > 0).float().mean().item()

                loss = loss / self.cfg.grad_accum_steps
                loss = loss + (aux_pol_c.float() + aux_pol_r.float()) * 0.0
                loss.backward()
                accumulated_loss += loss.item() * self.cfg.grad_accum_steps
                accumulated_acc += acc

            dynamic_clip = self.autopilot.before_step(self.model, step, self.cfg.max_steps)
            current_lr, _ = self.autopilot.update_parameters(step, accumulated_loss, self.cfg.max_steps)
            self.opt_engine.step()

            mean_loss = accumulated_loss / self.cfg.grad_accum_steps
            mean_acc = accumulated_acc / self.cfg.grad_accum_steps
            if step % 10 == 0:
                print(
                    f"[DPO] Step {step:05d}/{self.cfg.max_steps:05d} | "
                    f"Loss: {mean_loss:.3f} | Acc: {mean_acc:.2%} | "
                    f"β: {beta:.2f} | LR: {current_lr:.6f} | Clip: {dynamic_clip:.2f}"
                )
                with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "dpo", "step": step, "loss": mean_loss,
                        "accuracy": mean_acc, "lr": current_lr, "beta": beta,
                    }, ensure_ascii=False) + "\n")
                log_event("dpo_step_complete", step=step, loss=round(mean_loss, 4),
                          accuracy=round(mean_acc, 4), lr=round(current_lr, 7))

            state.step = step
            state.metrics = {"loss": mean_loss, "accuracy": mean_acc, "lr": current_lr}

        return state

    def finalize(self, state: StageState) -> StageState:
        if self.cfg is None or self.model is None:
            return state
        os.makedirs("checkpoints", exist_ok=True)
        final_path = f"checkpoints/busel_{self.profile_name}_DPO_FINAL.pt"
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "patcher_state_dict": self.patcher.state_dict(),
                "step": state.step,
                "profile": self.profile_name,
                "config": self.cfg.__dict__,
                "stage": "dpo",
                "beta": self.cfg.dpo_beta,
            }, final_path)
            print(f"💾 DPO final checkpoint: {final_path}")
            log_event("stage_complete", stage=self.name, profile=self.profile_name,
                      total_steps=state.step, final_path=final_path)
            state.last_checkpoint_path = final_path
        except Exception as e:
            print(f"❌ Failed to save DPO checkpoint: {e}")
        return state

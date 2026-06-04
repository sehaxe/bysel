"""
🛰️ busel EVAL STAGE v1.0 — Pipeline stage that runs all eval metrics
Loads a checkpoint, runs `tools.eval.run_all_metrics`, stashes the result
dict in `state.artifact` for downstream consumers (Telegram bot, web
dashboard, or just the orchestrator log).
"""
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any

import torch
import yaml

from training.stages.base import BaseStage, StageState, register_stage
from busel_logging import setup_logging, log_event


@dataclass
class buselEvalConfig:
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
    sft_glob: str = "data_train/sft/**/*.jsonl"
    max_perplexity_samples: int = 16
    max_sft_examples: int = 8
    max_format_prompts: int = 4

    @classmethod
    def from_profile(cls, profile_dict: dict, stage_params: dict | None = None) -> "buselEvalConfig":
        cfg = cls()
        sp = dict(stage_params or {})
        m = profile_dict.get("model", {})
        cfg.d_model = int(m.get("d_model", cfg.d_model))
        cfg.n_layers = int(m.get("n_layers", cfg.n_layers))
        cfg.n_heads = int(m.get("n_heads", cfg.n_heads))
        cfg.expert_hidden = int(m.get("expert_hidden", cfg.expert_hidden))
        cfg.num_experts = int(m.get("num_experts", cfg.num_experts))
        cfg.top_k = int(m.get("top_k", cfg.top_k))
        cfg.vocab_size = int(m.get("vocab_size", cfg.vocab_size))
        cfg.n_hyper = int(m.get("n_hyper", cfg.n_hyper))
        cfg.data_path = sp.get("data_path", cfg.data_path)
        cfg.sft_glob = sp.get("sft_glob", cfg.sft_glob)
        cfg.max_perplexity_samples = int(sp.get("max_perplexity_samples", cfg.max_perplexity_samples))
        cfg.max_sft_examples = int(sp.get("max_sft_examples", cfg.max_sft_examples))
        cfg.max_format_prompts = int(sp.get("max_format_prompts", cfg.max_format_prompts))
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


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@register_stage("eval")
class buselEvalStage:
    """Eval stage. Loads a checkpoint, runs all metrics, stashes in state.artifact."""

    name: str = "eval"

    def __init__(self) -> None:
        self.cfg: buselEvalConfig | None = None
        self.profile_name: str = "shpak"
        self.device: str = "cpu"
        self.model: Any = None
        self.patcher: Any = None
        self.checkpoint_path: str | None = None

    def setup(
        self,
        profile: dict | str,
        profile_name: str = "shpak",
        *,
        resume: str | None = None,
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

        self.cfg = buselEvalConfig.from_profile(profile_dict, stage_params)
        self.profile_name = profile_name
        self.checkpoint_path = resume

        setup_logging()
        log_event("eval_start", profile=profile_name, checkpoint=resume)
        self.device = _detect_device()

        from model.patching import StridedFastBLTPatcher
        from model.backbone import buselModel

        self.patcher = StridedFastBLTPatcher(d_model=self.cfg.d_model).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)

        if resume and os.path.exists(resume):
            ckpt = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
            self.patcher.load_state_dict(_strip_compile_prefix(ckpt["patcher_state_dict"]))
            print(f"📥 Loaded checkpoint for eval: {resume}")
        else:
            raise FileNotFoundError(
                f"Eval stage requires `resume=` to point to a checkpoint. Got: {resume!r}"
            )
        self.model.eval()
        self.patcher.eval()

    def run(self, state: StageState) -> StageState:
        if self.cfg is None or self.model is None:
            raise RuntimeError("setup() must be called before run()")
        from tools.eval import run_all_metrics
        print(f"🛰️  Running eval suite on {self.checkpoint_path}...")
        t0 = time.time()
        results = run_all_metrics(
            self.model, self.patcher,
            data_path=self.cfg.data_path,
            sft_glob=self.cfg.sft_glob,
            device=self.device,
            max_perplexity_samples=self.cfg.max_perplexity_samples,
            max_sft_examples=self.cfg.max_sft_examples,
            max_format_prompts=self.cfg.max_format_prompts,
        )
        elapsed = time.time() - t0
        results["eval_elapsed_s"] = round(elapsed, 2)
        results["checkpoint"] = self.checkpoint_path
        print(f"🛰️  Eval done in {elapsed:.1f}s:")
        for k, v in results.items():
            print(f"   {k}: {v}")
        state.artifact = results
        state.metrics = results
        log_event("eval_complete", **results)
        os.makedirs("checkpoints", exist_ok=True)
        with open(
            os.path.join("checkpoints", f"eval_{self.profile_name}_{int(time.time())}.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return state

    def finalize(self, state: StageState) -> StageState:
        if state.artifact is None:
            return state
        log_event("stage_complete", stage=self.name, profile=self.profile_name,
                  **{k: v for k, v in state.artifact.items() if isinstance(v, (int, float, str, bool))})
        return state

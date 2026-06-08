"""
📚 busel DATA PRESETS v1.0 — Named data catalog for SFT/DPO/eval stages

A static catalog of HF datasets and custom generators indexed by preset name.
This is NOT a plug-in extension point (no @register) — it's a simple dict
that the download_preset command reads at runtime.

To add a new preset:
1. Add an entry to PRESETS with name, stage, and either hf_dataset (for HF
   streaming) or generator_script (for custom generation).
2. Add a format adapter in the downloader if the field layout is non-standard.
3. The CLI auto-discovers it: `uv run cli.py download-preset --name <name>`.
"""
from __future__ import annotations
from typing import Any, Optional


STAGE_SFT = "sft"
STAGE_DPO = "dpo"
STAGE_EVAL = "eval"


FMT_CHAT_MESSAGES = "chat_messages"
FMT_PROMPT_CHOSEN_REJECTED = "prompt_chosen_rejected"
FMT_INSTRUCTION_INPUT_OUTPUT = "instruction_input_output"
FMT_CODE_PROBLEM_SOLUTION = "code_problem_solution"
FMT_TOOL_CALL = "tool_call"
FMT_CONVERSATIONS = "conversations"  # HF 'conversations' field: list[{from, value}]


PRESETS: dict[str, dict[str, Any]] = {
    "sft-shpak-chat": {
        "stage": STAGE_SFT,
        "hf_dataset": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "limit": 5000,
        "format_adapter": FMT_CHAT_MESSAGES,
        "output_subdir": "sft/chat",
        "description": "Multi-turn chat from UltraChat 200k (English)",
        "license": "CC BY-NC 4.0",
    },
    "sft-shpak-code": {
        "stage": STAGE_SFT,
        "hf_dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "split": "train",
        "limit": 3000,
        "format_adapter": FMT_INSTRUCTION_INPUT_OUTPUT,
        "output_subdir": "sft/code",
        "description": "Code generation from Magicoder OSS Instruct (75k English)",
        "license": "CC BY-SA 4.0",
    },
    "sft-shpak-tools": {
        "stage": STAGE_SFT,
        "hf_dataset": "Salesforce/xlam-function-calling-60k",
        "split": "train",
        "limit": 3000,
        "format_adapter": FMT_TOOL_CALL,
        "output_subdir": "sft/tools",
        "description": "Tool use / function calling from xLAM (60k English)",
        "license": "CC BY-NC 4.0",
    },
    "dpo-shpak": {
        "stage": STAGE_DPO,
        "hf_dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "limit": 5000,
        "format_adapter": FMT_PROMPT_CHOSEN_REJECTED,
        "output_subdir": "dpo/general",
        "description": "General preference pairs from UltraFeedback (English)",
        "license": "CC BY-NC 4.0",
    },
    "sft-shpak-glm-reasoning": {
        "stage": STAGE_SFT,
        "hf_dataset": "Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
        "split": "train",
        "limit": 50000,
        "format_adapter": FMT_CONVERSATIONS,
        "output_subdir": "sft/reasoning",
        "description": "Chain-of-thought reasoning from GLM-5.1 (746K English, CoT with <think> tags)",
        "license": "Apache 2.0",
    },
    "sft-shpak-claude-trace": {
        "stage": STAGE_SFT,
        "hf_dataset": "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        "split": "train",
        "limit": 5000,
        "format_adapter": FMT_CONVERSATIONS,
        "output_subdir": "sft/trace",
        "description": "Claude Opus 4.7 trace inversion — inverted CoT + clean answers (5K English)",
        "license": "Apache 2.0",
    },
}


def list_presets(stage: Optional[str] = None) -> list[str]:
    """Return sorted preset names. Optionally filter by stage (sft/dpo/eval)."""
    if stage is None:
        return sorted(PRESETS.keys())
    return sorted(name for name, meta in PRESETS.items() if meta.get("stage") == stage)


def get_preset(name: str) -> dict[str, Any]:
    """Look up a preset by name. Raises KeyError with helpful message on miss."""
    if name not in PRESETS:
        available = list_presets()
        raise KeyError(
            f"Unknown preset: {name!r}. Available: {available}"
        )
    return PRESETS[name]


def resolve_preset(preset_name: str, override_limit: Optional[int] = None) -> dict[str, Any]:
    """Resolve a preset into a concrete download plan.

    Returns a dict with all fields populated and `limit` applied (override
    takes precedence over the preset's default).

    Raises:
        KeyError: preset name is unknown.
        RuntimeError: preset depends on a generator script that hasn't been
            generated yet (Phase 4 stub).
    """
    meta = get_preset(preset_name)
    plan = dict(meta)
    if override_limit is not None:
        plan["limit"] = override_limit
    if plan.get("hf_dataset") is None and plan.get("generator_script"):
        raise RuntimeError(
            f"Preset {preset_name!r} is a generator stub. Run "
            f"`uv run cli.py generate-safety-data --output <path>` first, "
            f"or import {plan['generator_script']} directly."
        )
    return plan


__all__ = [
    "PRESETS",
    "STAGE_SFT",
    "STAGE_DPO",
    "STAGE_EVAL",
    "FMT_CHAT_MESSAGES",
    "FMT_PROMPT_CHOSEN_REJECTED",
    "FMT_INSTRUCTION_INPUT_OUTPUT",
    "FMT_CODE_PROBLEM_SOLUTION",
    "FMT_TOOL_CALL",
    "FMT_CONVERSATIONS",
    "list_presets",
    "get_preset",
    "resolve_preset",
]

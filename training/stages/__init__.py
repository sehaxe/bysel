"""
🛸 busel STAGES v1.0 — Multi-Stage Training Pipeline
Plug-in extension points for training stages (pretrain, SFT, DPO, eval, REPL).

Each stage is registered via @register_stage("name") and implements the
BaseStage Protocol. The pipeline orchestrator in tools/orchestrator.py
loads a configs/pipelines/<name>.yaml and runs the stages sequentially.
"""
from training.stages.base import (
    BaseStage,
    StageState,
    StageSpec,
    PipelineConfig,
    register_stage,
    get_stage,
    list_stages,
    is_stage_registered,
    load_pipeline_yaml,
)

from training.stages import pretrain as _pretrain_module  # noqa: F401  (triggers @register_stage)
from training.stages import sft as _sft_module  # noqa: F401  (triggers @register_stage)
from training.stages import dpo as _dpo_module  # noqa: F401  (triggers @register_stage)
from training.stages import eval as _eval_module  # noqa: F401  (triggers @register_stage)

__all__ = [
    "BaseStage",
    "StageState",
    "StageSpec",
    "PipelineConfig",
    "register_stage",
    "get_stage",
    "list_stages",
    "is_stage_registered",
    "load_pipeline_yaml",
]

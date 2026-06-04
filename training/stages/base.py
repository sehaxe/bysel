"""
🛸 busel STAGE BASE v1.0 — Protocol + State + Registry Helpers
A single training stage encapsulates one phase (pretrain, SFT, DPO, eval,
REPL). Stages are registered via @register_stage("name") and run by the
pipeline orchestrator in tools/orchestrator.py.

Stage lifecycle:
    setup(cfg) → run(state) → finalize(state)

- setup()  builds model, optimizer, dataloader from cfg
- run()    executes the training loop, may take hours/days
- finalize() saves the final checkpoint and logs the stage_complete event

Per-stage state (step, metrics, last checkpoint) flows in a StageState
dataclass that is passed between run() calls so the pipeline orchestrator
can chain stages and resume from the last checkpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, runtime_checkable

import yaml

import busel_registry as _reg_mod
from busel_registry import register as _register_outer


@dataclass
class StageState:
    """Mutable per-stage state, threaded through the pipeline.

    Fields:
        step: Current global step within this stage.
        epoch: Current epoch (0-indexed).
        best_loss: Best loss observed so far in this stage.
        metrics: Free-form metrics dict (loss, lr, moe_aux, grad_norm, ...).
        last_checkpoint_path: Absolute path to the most recent checkpoint.
        artifact: Stage-specific return value (e.g. eval results dict).
            Stages can stash anything here for downstream stages to consume.
    """

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    last_checkpoint_path: str | None = None
    artifact: Any = None


@dataclass
class StageSpec:
    """A single stage's config from pipeline.yaml.

    Fields:
        name: Stage name, must be registered in the "stage" registry
            (e.g. "pretrain", "sft", "dpo", "eval").
        data_preset: Optional data preset name. Resolved by the stage itself.
        resume: Optional path to a checkpoint to resume from.
        checkpoint_out: Optional path to write the final checkpoint to.
            If None, defaults to checkpoints/busel_<name>_FINAL.pt.
        params: Free-form dict, passed to stage.setup(cfg) for stage-specific
            tuning (max_steps, lr, beta, etc.).
    """

    name: str
    data_preset: str | None = None
    resume: str | None = None
    checkpoint_out: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """The full pipeline.yaml schema.

    Fields:
        name: Pipeline name (matches the YAML filename without .yaml).
        stages: Ordered list of stages to execute.
        global_params: Free-form dict applied to every stage's setup() call
            unless overridden per-stage.
    """

    name: str
    stages: list[StageSpec]
    global_params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseStage(Protocol):
    """Interface that every training stage must implement.

    Implementations register themselves with @register_stage("name") and
    provide concrete setup/run/finalize methods.
    """

    name: str

    def setup(self, cfg: Any) -> None:
        """Initialize model, optimizer, dataloader, etc. Called once.

        Args:
            cfg: buselConfig-like object (per-stage). Implementations
                typically receive a copy of global_params merged with the
                stage's own params from pipeline.yaml.
        """
        ...

    def run(self, state: StageState) -> StageState:
        """Execute the stage's training loop.

        Should mutate `state` in-place AND return it for convenience. The
        loop is bounded by stage-specific stop conditions (e.g. cfg.max_steps
        for pretrain, exhaustion of dataloader for SFT, fixed pairs for DPO).
        """
        ...

    def finalize(self, state: StageState) -> StageState:
        """Save final checkpoint, log stage_complete event, clean up.

        Default implementation just ensures state.last_checkpoint_path is
        set. Stages typically override to call torch.save() and emit a
        busel_logging.log_event("stage_complete", ...).
        """
        ...


def register_stage(name: str, *, override: bool = False):
    """Class decorator: @register_stage("pretrain").

    Equivalent to busel_registry.register("stage", name, override=override).
    Exists as a separate helper so consumers don't need to import both
    modules.
    """
    return _register_outer("stage", name, override=override)


def get_stage(name: str) -> type[BaseStage]:
    """Retrieve a stage class by name. Wraps busel_registry.get."""
    return _reg_mod.get("stage", name)  # type: ignore[no-any-return]


def list_stages() -> list[str]:
    """Sorted list of all registered stage names."""
    return _reg_mod.list_registered("stage")


def is_stage_registered(name: str) -> bool:
    """Boolean check without raising on misses."""
    return _reg_mod.is_registered("stage", name)


def load_pipeline_yaml(path: str | os.PathLike) -> PipelineConfig:
    """Load and validate a pipeline.yaml file.

    Schema (all keys required unless noted):
        name: <string>                   # pipeline name
        stages:                          # ordered list
          - name: <string>               # must be registered
            data_preset: <string|null>   # optional
            resume: <string|null>        # optional
            checkpoint_out: <string|null># optional
            params: {<key>: <value>}     # optional
        global_params: {<key>: <value>}  # optional

    Returns:
        PipelineConfig instance.

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        KeyError: if a required key is missing.
        ValueError: if a stage name is not registered.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipeline YAML not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline YAML root must be a dict, got {type(raw).__name__}")

    if "name" not in raw:
        raise KeyError(f"Pipeline YAML {p} missing required key 'name'")
    if "stages" not in raw:
        raise KeyError(f"Pipeline YAML {p} missing required key 'stages'")
    if not isinstance(raw["stages"], list) or len(raw["stages"]) == 0:
        raise ValueError(f"Pipeline YAML {p} 'stages' must be a non-empty list")

    stages: list[StageSpec] = []
    for idx, s in enumerate(raw["stages"]):
        if not isinstance(s, dict):
            raise ValueError(f"Pipeline YAML {p} stages[{idx}] must be a dict")
        if "name" not in s:
            raise KeyError(f"Pipeline YAML {p} stages[{idx}] missing required key 'name'")
        if not is_stage_registered(s["name"]):
            raise ValueError(
                f"Pipeline YAML {p} stages[{idx}].name={s['name']!r} is not a "
                f"registered stage. Available: {list_stages() or '(none)'}"
            )
        stages.append(
            StageSpec(
                name=s["name"],
                data_preset=s.get("data_preset"),
                resume=s.get("resume"),
                checkpoint_out=s.get("checkpoint_out"),
                params=dict(s.get("params", {})),
            )
        )

    return PipelineConfig(
        name=raw["name"],
        stages=stages,
        global_params=dict(raw.get("global_params", {})),
    )

"""
⚙️ busel ORCHESTRATOR v6.1 — Multi-Stage Pipeline
Содержит команды запуска обучения, автопилота, профайлера, и pipeline.
"""

import os
import sys
import subprocess
import typer

DATA_DIR = "data_train"


def load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def print_tui_header():
    typer.echo(typer.style("╔═══════════════════════════════════════════════════════════════════════════╗", fg=typer.colors.MAGENTA, bold=True))
    typer.echo(typer.style("║                            busel OMNI-LLM v6.1                            ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("║                 Sovereign 1-bit Any-to-Text AI Framework                  ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("╚═══════════════════════════════════════════════════════════════════════════╝", fg=typer.colors.MAGENTA, bold=True))


def _build_shim_yaml(profile: str, resume: str, max_steps, warmup_steps) -> str:
    """Build a temp pipeline YAML (pretrain-only + overrides); return the temp dir path."""
    import tempfile
    import yaml as _yaml
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "pipelines", "pretrain-only.yaml")
    with open(src) as f:
        cfg = _yaml.safe_load(f)
    stage = cfg["stages"][0]
    stage.setdefault("params", {})
    if profile:
        stage["params"]["profile_name"] = profile
    if max_steps is not None:
        stage["params"]["max_steps"] = max_steps
    if warmup_steps is not None:
        stage["params"]["warmup_steps"] = warmup_steps
    if resume:
        stage["resume"] = resume
    tmpdir = tempfile.mkdtemp(prefix="busel_shim_")
    tmp_yaml = os.path.join(tmpdir, "shim.yaml")
    with open(tmp_yaml, "w") as f:
        _yaml.dump(cfg, f)
    return tmpdir


def train_single_profile(args_list):
    """Translate legacy train.py CLI args into a pipeline run.

    Supported: --profile, --resume, --max-steps, --warmup-steps.
    Other flags (--no-compile, --compile-mode, --no-checkpointing, --seed) are dropped
    because the pipeline runner owns those knobs (see configs/pipelines/*.yaml).
    """
    import argparse
    import shutil
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile", "-p", default="shpak")
    p.add_argument("--resume", "-r", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    args, _unknown = p.parse_known_args(args_list)

    tmpdir = _build_shim_yaml(args.profile, args.resume, args.max_steps, args.warmup_steps)
    try:
        pipeline(name="shim", start_stage=None, config_dir=tmpdir)
        return 0
    except SystemExit as e:
        return e.code or 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def autopilot(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile name: shpak or zubr")
):
    print_tui_header()
    load_env()

    want_monitoring = typer.confirm("📊 Do you want to enable local logging & TensorBoard monitoring?", default=True)
    if want_monitoring:
        typer.echo(typer.style("📈 Monitoring activated. Run 'tensorboard --logdir=checkpoints' to view logs.\n", fg=typer.colors.GREEN))

    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        typer.echo(typer.style("📁 Directory 'data_train' is empty. Starting automatic download...", fg=typer.colors.YELLOW, bold=True))
        from tools.data_manager import _download_text, _download_sft, _download_vision
        _download_text(80000, "smollm")
        _download_sft(5000, "smoltalk")
        _download_vision(500, "HuggingFaceM4/COCO")
    else:
        typer.echo(typer.style("📁 Training data found. Skipping download.", fg=typer.colors.GREEN))

    typer.echo(typer.style("\n📊 Launching hardware express-profiler for MPS/CUDA testing...", fg=typer.colors.CYAN, bold=True))
    result = subprocess.run([sys.executable, "tests/profiler_run.py"])
    if result.returncode != 0:
        typer.echo(typer.style("❌ Hardware test failed! Please check your GPU/accelerator.", fg=typer.colors.RED, bold=True))
        raise typer.Exit(code=1)

    typer.echo("=" * 80)

    typer.echo(typer.style(f"🔥 AUTOPILOT: Launching main training loop [{profile_name.upper()}]...", fg=typer.colors.GREEN, bold=True))
    train_single_profile(["--profile", profile_name])


def train(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile: shpak or zubr"),
    resume: str = typer.Option(None, "--resume", "-r", help="Path to checkpoint for resuming")
):
    args = ["--profile", profile_name]
    if resume:
        args.extend(["--resume", resume])
    train_single_profile(args)


def train_all(
    start_stage: str = typer.Option(None, "--start-stage", help="Resume from this stage name (e.g. 'sft', 'dpo')"),
):
    """🚀 ONE-CLICK FULL TRAINING: pretrain → SFT → DPO → eval.

    Runs the `full` pipeline (configs/pipelines/full.yaml). Requires that
    the 4 HF data presets are already downloaded — run
    `uv run cli.py download-data` first.
    """
    pipeline(name="full", start_stage=start_stage, config_dir="configs/pipelines")


def profile():
    subprocess.run([sys.executable, "tests/profiler_run.py"])


def pipeline(
    name: str = typer.Option(..., "--name", "-n", help="Pipeline name (configs/pipelines/<name>.yaml)"),
    start_stage: str = typer.Option(None, "--start-stage", help="Resume from this stage name"),
    config_dir: str = typer.Option("configs/pipelines", "--config-dir", help="Where to look for pipeline YAMLs"),
):
    """Run a multi-stage training pipeline.

    Loads configs/pipelines/<name>.yaml, instantiates each registered
    stage via training/stages, and runs setup → run → finalize in order.
    Per-stage checkpoints are saved automatically.
    """
    from training.stages import load_pipeline_yaml, get_stage
    from busel_logging import setup_logging, log_event
    from training.stages.base import StageState

    print_tui_header()
    load_env()
    setup_logging()

    yaml_path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(yaml_path):
        typer.echo(typer.style(f"❌ Pipeline YAML not found: {yaml_path}", fg=typer.colors.RED, bold=True))
        typer.echo(typer.style(f"   Available pipelines in {config_dir}:", fg=typer.colors.YELLOW))
        if os.path.isdir(config_dir):
            for f in sorted(os.listdir(config_dir)):
                if f.endswith(".yaml"):
                    typer.echo(f"     - {f[:-5]}")
        raise typer.Exit(code=1)

    pipeline_cfg = load_pipeline_yaml(yaml_path)
    log_event("pipeline_start", pipeline=pipeline_cfg.name, num_stages=len(pipeline_cfg.stages))

    typer.echo(typer.style(f"🛸 Pipeline: {pipeline_cfg.name} ({len(pipeline_cfg.stages)} stages)", fg=typer.colors.CYAN, bold=True))
    for i, s in enumerate(pipeline_cfg.stages, 1):
        typer.echo(typer.style(f"   {i}. {s.name}  data={s.data_preset or '-'}  resume={s.resume or '-'}", fg=typer.colors.CYAN))

    import yaml as _yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        _default_profiles = _yaml.safe_load(f).get("profiles", {})

    def _resolve_resume(stage_name: str, default_resume: str | None) -> str | None:
        if default_resume:
            return default_resume
        candidate = f"checkpoints/busel_{pipeline_cfg.name}_{stage_name}_FINAL.pt"
        return candidate if os.path.exists(candidate) else None

    state = StageState()
    if not isinstance(start_stage, str):
        start_stage = None
    skipping = bool(start_stage)
    running_resume: str | None = None

    for i, stage_spec in enumerate(pipeline_cfg.stages, 1):
        if skipping:
            if stage_spec.name == start_stage:
                skipping = False
            else:
                typer.echo(typer.style(f"⏭  Skipping stage {i}/{len(pipeline_cfg.stages)}: {stage_spec.name}", fg=typer.colors.YELLOW))
                continue

        typer.echo(typer.style(f"\n🚀 Stage {i}/{len(pipeline_cfg.stages)}: {stage_spec.name}", fg=typer.colors.GREEN, bold=True))
        log_event("stage_start", pipeline=pipeline_cfg.name, stage=stage_spec.name, index=i)

        stage_cls = get_stage(stage_spec.name)
        stage = stage_cls()

        merged_params = {**pipeline_cfg.global_params, **stage_spec.params}
        if stage_spec.checkpoint_out and "checkpoint_out" not in merged_params:
            merged_params["checkpoint_out"] = stage_spec.checkpoint_out
        profile_name = merged_params.pop("profile_name", stage_spec.data_preset or "shpak")
        profile_dict = _default_profiles.get(profile_name)
        if profile_dict is None:
            raise ValueError(f"Profile {profile_name!r} not in configs/default.yaml")

        resume = _resolve_resume(stage_spec.name, stage_spec.resume)
        if running_resume and stage_spec.name != "pretrain" and not stage_spec.resume:
            resume = running_resume

        try:
            stage.setup(
                profile=profile_dict,
                profile_name=profile_name,
                resume=resume,
                stage_params=merged_params,
            )
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} setup() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="setup", error=str(e))
            raise typer.Exit(code=1)

        try:
            state = stage.run(state)
        except SystemExit:
            raise
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} run() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="run", error=str(e))
            raise typer.Exit(code=1)

        try:
            state = stage.finalize(state)
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} finalize() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="finalize", error=str(e))
            raise typer.Exit(code=1)

    if state.last_checkpoint_path:
        running_resume = state.last_checkpoint_path

    log_event("pipeline_complete", pipeline=pipeline_cfg.name, total_stages=len(pipeline_cfg.stages))
    typer.echo(typer.style(f"\n🎉 Pipeline {pipeline_cfg.name} complete! {len(pipeline_cfg.stages)} stages succeeded.", fg=typer.colors.GREEN, bold=True))


PROFILE_LADDER = ["chyzh", "shpak", "zubr"]
PROFILE_PARAMS = {"chyzh": 10_000_000, "shpak": 55_000_000, "zubr": 120_000_000}
CHINCHILLA = {p: 80 * n for p, n in PROFILE_PARAMS.items()}


def _load_profile_block(profile: str) -> dict:
    import yaml as _yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        full = _yaml.safe_load(f)
    return full["profiles"][profile]


def plan_escalation(target: str, max_steps: int | None = None, vram_gb: float = 16.0, chin_cap: float = 1.5) -> dict:
    """🪜 Smart escalation planner: derive ladder + per-stage config from target.

    Default semantics: "as long as possible" = Chinchilla-optimal max_steps.
    Optional: pass --max-steps N to cap total training across all stages.

    Scaling laws applied:
      - Chinchilla: D_pretrain = 80 × N_params bytes
      - Time per step ∝ batch × ctx × params (linear in tokens × model size)
      - max_steps capped at chin_cap × Chinchilla steps to avoid overtraining small models
      - batch_size, chunk_size, step_ms_est all read from `configs/default.yaml` profile
        (no hardcoded safety caps — user owns the config)
    """
    if target not in PROFILE_LADDER:
        raise ValueError(f"target must be one of {PROFILE_LADDER}, got {target!r}")

    ladder = [target]
    n = len(ladder)

    per_stage_cap = None
    if max_steps is not None:
        per_stage_cap = max(100, max_steps // n)

    stages = []
    for profile in ladder:
        prof = _load_profile_block(profile)
        chunk_size = int(prof["data"]["chunk_size"])
        batch_size = int(prof["data"]["batch_size"])
        step_ms = int(prof.get("perf", {}).get("step_ms_est", 250))
        tokens_per_step = batch_size * (chunk_size // 4)
        chin_tokens = CHINCHILLA[profile]
        chin_steps = int(chin_tokens / tokens_per_step)
        cap_steps = int(chin_cap * chin_steps)
        candidates = [cap_steps]
        if per_stage_cap is not None:
            candidates.append(per_stage_cap)
        max_steps_actual = min(candidates)
        chin_pct = 100.0 * (max_steps_actual * tokens_per_step) / chin_tokens
        est_h = max_steps_actual * step_ms / 1000 / 3600
        stages.append(
            {
                "profile": profile,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "max_steps": max_steps_actual,
                "est_h": est_h,
                "step_ms": step_ms,
                "chinchilla_pct": chin_pct,
            }
        )

    return {"target": target, "ladder": ladder, "stages": stages, "vram_gb": vram_gb, "max_steps": max_steps}


def _write_escalation_yaml(plan: dict, path: str) -> None:
    import yaml as _yaml

    stages_yaml = []
    for s in plan["stages"]:
        stages_yaml.append(
            {
                "name": "pretrain",
                "data_preset": s["profile"],
                "checkpoint_out": f"checkpoints/busel_escalate_{s['profile']}_FINAL.pt",
                "params": {
                    "profile_name": s["profile"],
                    "max_steps": s["max_steps"],
                    "warmup_steps": max(50, int(0.05 * s["max_steps"])),
                    "batch_size": s["batch_size"],
                    "chunk_size": s["chunk_size"],
                },
            }
        )
    cap_note = f"max_steps={plan['max_steps']}" if plan.get("max_steps") else "as long as possible (Chinchilla)"
    pipeline_dict = {
        "name": f"escalate-{plan['target']}",
        "description": f"Smart escalation: {' → '.join(plan['ladder'])} | {cap_note}",
        "stages": stages_yaml,
        "global_params": {},
    }
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(pipeline_dict, f, sort_keys=False, allow_unicode=True)


def _print_escalation_plan(plan: dict) -> None:
    typer.echo(typer.style("\n🪜 SMART ESCALATION PLAN", fg=typer.colors.CYAN, bold=True))
    cap_text = f"max_steps={plan['max_steps']}" if plan.get("max_steps") else "as long as possible (Chinchilla)"
    typer.echo(typer.style(f"   target: {plan['target']} | ladder: {' → '.join(plan['ladder'])} | VRAM={plan['vram_gb']:.0f} GB | {cap_text}", fg=typer.colors.CYAN))
    typer.echo("")
    typer.echo(f"   {'profile':>8} {'est_h':>7} {'max_steps':>11} {'batch':>6} {'chunk':>6} {'chin_%':>7}")
    for s in plan["stages"]:
        typer.echo(
            f"   {s['profile']:>8} {s['est_h']:>7.2f} {s['max_steps']:>11,} {s['batch_size']:>6} {s['chunk_size']:>6} {s['chinchilla_pct']:>6.1f}%"
        )
    typer.echo("")


def escalate(
    target: str = typer.Option("shpak", "--target", "-t", help="Target profile: chyzh | shpak | zubr"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Cap total training across all stages (default: train to Chinchilla)"),
    vram_gb: float = typer.Option(16.0, "--vram", help="Available VRAM in GB (clamps batch_size)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan only, do not execute"),
):
    print_tui_header()
    plan = plan_escalation(target, max_steps=max_steps, vram_gb=vram_gb)
    _print_escalation_plan(plan)
    if dry_run:
        typer.echo(typer.style("🏁 Dry-run complete. No execution.", fg=typer.colors.YELLOW))
        return
    yaml_name = f".escalate-{target}-{max_steps}" if max_steps else f".escalate-{target}-chinchilla"
    yaml_path = f"configs/pipelines/{yaml_name}.yaml"
    _write_escalation_yaml(plan, yaml_path)
    typer.echo(typer.style(f"📝 Plan persisted to {yaml_path}", fg=typer.colors.GREEN))
    typer.echo(typer.style("🚀 Handing off to pipeline()...\n", fg=typer.colors.GREEN, bold=True))
    pipeline(name=yaml_name, config_dir="configs/pipelines")
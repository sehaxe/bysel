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
    subprocess.run([sys.executable, "train.py", "--profile", profile_name])


def train(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile: shpak or zubr"),
    resume: str = typer.Option(None, "--resume", "-r", help="Path to checkpoint for resuming")
):
    args = [sys.executable, "train.py", "--profile", profile_name]
    if resume:
        args.extend(["--resume", resume])
    subprocess.run(args)


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

    state = StageState()
    skipping = bool(start_stage)

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
        profile_name = merged_params.pop("profile_name", stage_spec.data_preset or "shpak")
        try:
            if stage_spec.name == "pretrain":
                import yaml as _yaml
                with open("configs/default.yaml", "r", encoding="utf-8") as f:
                    full = _yaml.safe_load(f)
                profile_dict = full["profiles"].get(profile_name)
                if profile_dict is None:
                    raise ValueError(f"Profile {profile_name!r} not in configs/default.yaml")
                profile_dict = {**profile_dict, "training": {**profile_dict.get("training", {}), **merged_params}}
                stage.setup(
                    profile=profile_dict,
                    profile_name=profile_name,
                    resume=stage_spec.resume,
                )
            else:
                stage.setup(merged_params)
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

    log_event("pipeline_complete", pipeline=pipeline_cfg.name, total_stages=len(pipeline_cfg.stages))
    typer.echo(typer.style(f"\n🎉 Pipeline {pipeline_cfg.name} complete! {len(pipeline_cfg.stages)} stages succeeded.", fg=typer.colors.GREEN, bold=True))
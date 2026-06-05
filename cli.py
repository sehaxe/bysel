"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                            BUSEL OMNI-LLM v5.5                            ║
║                 Sovereign 1-bit Any-to-Text AI Framework                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
import os

import typer
from tools.data_manager import download_all, download_vision, download_text, download_sft, label_vision, download_multimodal, download_preset, list_presets_cmd, download_data
from tools.orchestrator import autopilot, train, train_all, profile, pipeline, escalate

app = typer.Typer(
    help="busel Master CLI Engine - Sovereign 1-bit Omni-LLM",
    rich_markup_mode="markdown",
    invoke_without_command=True,
)


@app.callback()
def _default_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo(typer.style("🪜 No subcommand → defaulting to `escalate --target shpak`", fg=typer.colors.CYAN, bold=True))
        typer.echo(typer.style("   (batch/chunk/LR come from configs/default.yaml — edit there, not in code)\n", fg=typer.colors.CYAN))
        escalate(target="shpak")


# 📥 Регистрация команд подготовки данных
app.command(name="download-all", help="📥 Bulk download and prepare ALL standard datasets (Text, SFT, Vision) at once.")(download_all)
app.command(name="download-vision", help="📥 Stream and convert a ready-made vision dataset (COCO) from Hugging Face.")(download_vision)
app.command(name="download-text", help="📥 Stream and convert text pretrain datasets (TinyStories/FineWeb-Edu) from Hugging Face.")(download_text)
app.command(name="download-sft", help="📥 Download and prepare English instruction-following dataset (Alpaca).")(download_sft)
app.command(name="download-preset", help="📚 Download a named data preset (SFT/DPO) — see `cli.py list-presets`.")(download_preset)
app.command(name="list-presets", help="📚 List all available data presets (SFT/DPO).")(list_presets_cmd)
app.command(name="download-data", help="📥 Download EVERY data preset in one shot (3 SFT + 1 DPO HF datasets).")(download_data)
app.command(name="download-multimodal", help="🛰️ Generate synthetic image/video/audio/docx test files for the multimodal encoders (no internet).")(download_multimodal)
app.command(name="label-vision", help="🤖 Auto-label a local directory of images using a local Ollama vision model.")(label_vision)

# 🚀 Регистрация команд обучения и сервисов
app.command(name="autopilot", help="🛸 ULTIMATE ONE-CLICK AUTOPILOT: Verifies env, downloads data, profiles hardware, and launches training.")(autopilot)
app.command(name="train", help="🔥 Manually start the core training loop (single-stage, legacy).")(train)
app.command(name="train-all", help="🚀 ONE-CLICK FULL TRAINING: pretrain → SFT → DPO → eval (requires `download-data` first).")(train_all)
app.command(name="pipeline", help="🛸 Run a multi-stage training pipeline (configs/pipelines/<name>.yaml).")(pipeline)
app.command(name="escalate", help="🪜 Smart auto-escalation: chyzh→shpak→zubr with optimal config from scaling laws.")(escalate)


@app.command(name="stop", help="🛑 Graceful stop: create /tmp/busel_stop so training saves and exits at next step boundary.")
def stop_cmd():
    """🛑 Trigger a graceful save-and-exit for any running training run.

    The pretrain stage checks BUSEL_STOP_FILE (default /tmp/busel_stop) at the
    top of every step. If present, it logs the stop event, sets state.step to
    the current step, and returns from run() — which lets finalize() save the
    final checkpoint normally. No Ctrl+C required.
    """
    import pathlib
    path = os.environ.get("BUSEL_STOP_FILE", "/tmp/busel_stop")
    pathlib.Path(path).touch()
    typer.echo(typer.style(f"✅ Stop signal sent: {path}", fg=typer.colors.GREEN, bold=True))
    typer.echo("   Training will save FINAL checkpoint and exit at the next step boundary.")
app.command(name="profile", help="📊 Run the ultra-stable step-by-step performance profiler (v2.0) on Mac/CUDA.")(profile)

# 💬 НОВАЯ КОМАНДА: Локальный интерактивный чат
@app.command(name="chat", help="💬 Start interactive console chat with a trained model (auto-detects architecture).")
def chat(
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Path to .pt checkpoint (auto-detects latest if omitted)"),
    profile: str = typer.Option(None, "--profile", "-p", help="Force profile from configs/default.yaml"),
    device: str = typer.Option(None, "--device", "-d", help="Force device: cuda / mps / cpu"),
):
    import subprocess
    import sys

    args = [sys.executable, "tools/inference.py"]
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    if profile:
        args.extend(["--profile", profile])
    if device:
        args.extend(["--device", device])

    subprocess.run(args)

if __name__ == "__main__":
    app()
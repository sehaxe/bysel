"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                            BUSEL OMNI-LLM v5.5                            ║
║                 Sovereign 1-bit Any-to-Text AI Framework                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
import typer
from tools.data_manager import download_all, download_vision, download_text, download_sft, label_vision, download_multimodal
from tools.orchestrator import autopilot, train, profile, pipeline

app = typer.Typer(
    help="busel Master CLI Engine - Sovereign 1-bit Omni-LLM",
    rich_markup_mode="markdown"
)

# 📥 Регистрация команд подготовки данных
app.command(name="download-all", help="📥 Bulk download and prepare ALL standard datasets (Text, SFT, Vision) at once.")(download_all)
app.command(name="download-vision", help="📥 Stream and convert a ready-made vision dataset (COCO) from Hugging Face.")(download_vision)
app.command(name="download-text", help="📥 Stream and convert text pretrain datasets (TinyStories/FineWeb-Edu) from Hugging Face.")(download_text)
app.command(name="download-sft", help="📥 Download and prepare English instruction-following dataset (Alpaca).")(download_sft)
app.command(name="download-multimodal", help="🛰️ Generate synthetic image/video/audio/docx test files for the multimodal encoders (no internet).")(download_multimodal)
app.command(name="label-vision", help="🤖 Auto-label a local directory of images using a local Ollama vision model.")(label_vision)

# 🚀 Регистрация команд обучения и сервисов
app.command(name="autopilot", help="🛸 ULTIMATE ONE-CLICK AUTOPILOT: Verifies env, downloads data, profiles hardware, and launches training.")(autopilot)
app.command(name="train", help="🔥 Manually start the core training loop (single-stage, legacy).")(train)
app.command(name="pipeline", help="🛸 Run a multi-stage training pipeline (configs/pipelines/<name>.yaml).")(pipeline)
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
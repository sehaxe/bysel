"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                            BYSEL OMNI-LLM v4.0                            ║
║                 Sovereign 1-bit Any-to-Text AI Framework                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import typer
from tools.data_manager import download_all, download_vision, download_text, download_sft, label_vision
from tools.orchestrator import autopilot, train, profile, serve, bot
from telegram_bot.setup_wizard import run_setup_wizard



app = typer.Typer(
    help="Bysel Master CLI Engine - Sovereign 1-bit Omni-LLM",
    rich_markup_mode="markdown"
)

# 📥 Регистрация команд подготовки данных
app.command(name="download-all", help="📥 Bulk download and prepare ALL standard datasets (Text, SFT, Vision) at once.")(download_all)
app.command(name="download-vision", help="📥 Stream and convert a ready-made vision dataset (COCO) from Hugging Face.")(download_vision)
app.command(name="download-text", help="📥 Stream and convert text pretrain datasets (TinyStories/FineWeb-Edu) from Hugging Face.")(download_text)
app.command(name="download-sft", help="📥 Download and prepare English instruction-following dataset (Alpaca).")(download_sft)
app.command(name="label-vision", help="🤖 Auto-label a local directory of images using a local Ollama vision model.")(label_vision)

# 🚀 Регистрация команд обучения и сервисов
app.command(name="autopilot", help="🛸 ULTIMATE ONE-CLICK AUTOPILOT: Verifies env, downloads data, profiles hardware, and launches training.")(autopilot)
app.command(name="train", help="🔥 Manually start the core training loop.")(train)
app.command(name="profile", help="📊 Run the ultra-stable step-by-step performance profiler (v2.0) on Mac/CUDA.")(profile)
app.command(name="serve", help="⚡ Start the high-performance FastAPI inference API server.")(serve)
app.command(name="bot", help="🤖 Start the sovereign Telegram Bot (requires TELEGRAM_BOT_TOKEN in .env).")(bot)
app.command(name="setup", help="🧙 Run interactive .env setup wizard")

def setup_command():
    run_setup_wizard(force=True)


if __name__ == "__main__":
    app()
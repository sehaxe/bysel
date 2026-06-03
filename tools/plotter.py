"""
📈 busel MODERN GOOGLE-STYLE PLOTTER v3.0
Генерирует минималистичные графики в стиле Google Cloud / Vertex AI.
"""

import os
import json
import numpy as np


def generate_report_plot(log_path="checkpoints/metrics.jsonl", output_path="checkpoints/training_report.png"):
    """
    Парсит JSONL-лог метрик и строит трехпанельный плоский график в стиле Google Material.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if not os.path.exists(log_path):
        return False

    steps, losses, aux_losses, speeds, lrs, vrams = [], [], [], [], [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                steps.append(data["step"])
                losses.append(data["loss"])
                aux_losses.append(data["aux_loss"])
                speeds.append(data["speed"])
                lrs.append(data["lr"])
                vrams.append(data.get("vram", 0.0))
            except Exception:
                continue

    if not steps:
        return False

    steps = np.array(steps)
    losses = np.array(losses)
    aux_losses = np.array(aux_losses)
    speeds = np.array(speeds)
    lrs = np.array(lrs)
    vrams = np.array(vrams)

    tokens_per_step = 2048
    cumulative_tokens = steps * tokens_per_step

    def ema(data, alpha=0.15):
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    loss_smoothed = ema(losses, alpha=0.2)
    aux_smoothed = ema(aux_losses, alpha=0.2)
    speed_smoothed = ema(speeds, alpha=0.2)

    # Чистый системный шрифт без засечек
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#202124'  # Google Dark Gray
    plt.rcParams['axes.labelcolor'] = '#5F6368'  # Google Medium Gray
    plt.rcParams['xtick.color'] = '#5F6368'
    plt.rcParams['ytick.color'] = '#5F6368'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 14), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")  # Чистый белый фон страницы для плоского дизайна

    # 1. ПАНЕЛЬ 1: LOSS CONVERGENCE (Google Blue & Google Red)
    ax1.set_facecolor("#FFFFFF")
    line_loss_raw = ax1.plot(steps, losses, color="#1A73E8", alpha=0.15, linewidth=1.0)
    line_loss_smooth = ax1.plot(steps, loss_smoothed, color="#1A73E8", linewidth=2.0, label="Total Loss")
    ax1.set_ylabel("Total Loss", color="#1A73E8", fontweight="medium", fontsize=9)
    ax1.tick_params(axis='y', labelcolor="#1A73E8")
    
    ax1_twin = ax1.twinx()
    line_aux_raw = ax1_twin.plot(steps, aux_losses, color="#EA4335", alpha=0.15, linewidth=1.0, linestyle="--")
    line_aux_smooth = ax1_twin.plot(steps, aux_smoothed, color="#EA4335", linewidth=1.5, linestyle="--", label="Aux Loss (MoE)")
    ax1_twin.set_ylabel("Aux Loss", color="#EA4335", fontweight="medium", fontsize=9)
    ax1_twin.tick_params(axis='y', labelcolor="#EA4335")
    
    ax1.set_title("Loss Convergence & Expert Balance", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_1 = line_loss_smooth + line_aux_smooth
    labels_1 = [l.get_label() for l in lines_1]
    ax1.legend(lines_1, labels_1, loc="upper right", frameon=False, fontsize=8.5)

    # 2. ПАНЕЛЬ 2: COMPUTATIONAL EFFICIENCY (Google Green & Google Purple)
    ax2.set_facecolor("#FFFFFF")
    line_speed_raw = ax2.plot(steps, speeds, color="#34A853", alpha=0.15, linewidth=1.0)
    line_speed_smooth = ax2.plot(steps, speed_smoothed, color="#34A853", linewidth=2.0, label="Throughput")
    ax2.set_ylabel("Throughput (tokens/s)", color="#34A853", fontweight="medium", fontsize=9)
    ax2.tick_params(axis='y', labelcolor="#34A853")

    ax2_twin = ax2.twinx()
    line_vram = ax2_twin.plot(steps, vrams, color="#9333EA", alpha=0.5, linewidth=1.2, linestyle=":", label="VRAM Allocated")
    ax2_twin.set_ylabel("VRAM Allocated (MB)", color="#9333EA", fontweight="medium", fontsize=9)
    ax2_twin.tick_params(axis='y', labelcolor="#9333EA")

    ax2.set_title("System Compute Throughput & Memory", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_2 = line_speed_smooth + line_vram
    labels_2 = [l.get_label() for l in lines_2]
    ax2.legend(lines_2, labels_2, loc="upper left", frameon=False, fontsize=8.5)

    # 3. ПАНЕЛЬ 3: OPTIMIZER PROFILE (Google Amber & Google Teal)
    ax3.set_facecolor("#FFFFFF")
    line_lr = ax3.plot(steps, lrs, color="#F9AB00", linewidth=1.8, linestyle="-.", label="Learning Rate")
    ax3.set_ylabel("Learning Rate", color="#F9AB00", fontweight="medium", fontsize=9)
    ax3.tick_params(axis='y', labelcolor="#F9AB00")
    ax3.set_xlabel("Training Steps", fontweight="medium", fontsize=9)

    ax3_twin = ax3.twinx()
    line_tokens = ax3_twin.plot(steps, cumulative_tokens / 1e3, color="#12B5CB", linewidth=1.2, linestyle=":", label="Processed Volume")
    ax3_twin.set_ylabel("Cumulative Volume (K tokens)", color="#12B5CB", fontweight="medium", fontsize=9)
    ax3_twin.tick_params(axis='y', labelcolor="#12B5CB")

    ax3.set_title("Learning Rate Decay & Cumulative Data", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_3 = line_lr + line_tokens
    labels_3 = [l.get_label() for l in lines_3]
    ax3.legend(lines_3, labels_3, loc="upper right", frameon=False, fontsize=8.5)

    # 🎯 МАТЕРИАЛЬНЫЙ СТИЛЬ ОСЕЙ И СЕТКИ (Google Cloud Flat Design):
    # Оставляем только тонкую нижнюю ось, убираем деления (ticks) для воздушности.
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Только горизонтальные тонкие линии сетки
        ax.yaxis.grid(True, color='#F1F3F4', linestyle='-', linewidth=1.0)
        ax.xaxis.grid(False)
        ax.tick_params(axis='both', which='both', length=0)
        
    for ax_t in [ax1_twin, ax2_twin, ax3_twin]:
        ax_t.spines['top'].set_visible(False)
        ax_t.spines['right'].set_visible(False)
        ax_t.spines['left'].set_visible(False)
        ax_t.spines['bottom'].set_visible(False)
        ax_t.tick_params(axis='both', which='both', length=0)

    # 📊 СБОР И РАСЧЕТ ДАННЫХ ДЛЯ КАРТОЧЕК ДАШБОРДА
    min_loss_val = np.min(losses)
    min_loss_step = steps[np.argmin(losses)]
    avg_speed_val = np.mean(speeds[10:]) if len(speeds) > 10 else np.mean(speeds)
    max_vram_val = np.max(vrams)
    total_tokens_val = cumulative_tokens[-1]

    # Создаем аккуратную подложку для дашборда на самом верху
    rect = patches.FancyBboxPatch(
        (0.06, 0.905), 0.88, 0.07,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        facecolor="#F8F9FA", edgecolor="#F1F3F4", linewidth=1.0,
        transform=fig.transFigure, figure=fig, zorder=-1
    )
    fig.patches.append(rect)

    # 🎯 ВЕРТИКАЛЬНО ВЫРАВНЕННЫЕ МИНИМАЛИСТИЧНЫЕ КОЛОНКИ СТАТИСТИКИ (Vertex AI Style)
    # Колонка 1: Min Loss
    fig.text(0.16, 0.955, "MINIMUM LOSS", fontsize=7.5, color="#5F6368", fontweight="bold", ha="center")
    fig.text(0.16, 0.932, f"{min_loss_val:.4f}", fontsize=13, color="#1A73E8", fontweight="bold", ha="center")
    fig.text(0.16, 0.915, f"at Step {min_loss_step}", fontsize=7.5, color="#70757A", ha="center")

    # Колонка 2: Speed
    fig.text(0.38, 0.955, "COMPUTE SPEED", fontsize=7.5, color="#5F6368", fontweight="bold", ha="center")
    fig.text(0.38, 0.932, f"{avg_speed_val:.1f} tok/s", fontsize=13, color="#34A853", fontweight="bold", ha="center")
    fig.text(0.38, 0.915, "MacBook Air / MPS", fontsize=7.5, color="#70757A", ha="center")

    # Колонка 3: Total Volume
    fig.text(0.62, 0.955, "CUMULATIVE VOLUME", fontsize=7.5, color="#5F6368", fontweight="bold", ha="center")
    fig.text(0.62, 0.932, f"{total_tokens_val / 1e3:.1f} Ktok", fontsize=13, color="#12B5CB", fontweight="bold", ha="center")
    fig.text(0.62, 0.915, "Byte Tokens Processed", fontsize=7.5, color="#70757A", ha="center")

    # Колонка 4: Peak Memory
    fig.text(0.84, 0.955, "PEAK MEMORY", fontsize=7.5, color="#5F6368", fontweight="bold", ha="center")
    fig.text(0.84, 0.932, f"{max_vram_val:.1f} MB", fontsize=13, color="#9333EA", fontweight="bold", ha="center")
    fig.text(0.84, 0.915, "Allocated on MPS", fontsize=7.5, color="#70757A", ha="center")

    plt.subplots_adjust(top=0.87, hspace=0.38)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches='tight')
    plt.close()
    return True
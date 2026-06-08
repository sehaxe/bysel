import os
import sys
import csv
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.pipeline import get_busel_dataloader
from model.backbone import buselModel
from model.patching import StridedFastBLTPatcher
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine
from training.stages.pretrain import buselPretrainConfig, _build_targets

RESULTS_DIR = Path("checkpoints/scaling_laws")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_model_and_opt(yaml_cfg, device):
    cfg = buselPretrainConfig.from_profile(yaml_cfg)
    cfg.selective_backward = False
    cfg.backward_ratio = 1.0

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in patcher.parameters())

    loss_engine = buselLossEngine(cfg.vocab_size)
    opt_engine = buselOptimizerEngine(
        model, patcher,
        lr_muon=cfg.learning_rate_muon,
        lr_adamw=cfg.learning_rate_adamw,
        optimizer_type=cfg.optimizer_type,
        lotus_rank=cfg.lotus_rank,
        lotus_lr_scale=cfg.lotus_lr_scale,
        lr_multipliers=cfg.lr_multipliers,
        use_schedule_free=cfg.use_schedule_free,
        sf_beta=cfg.sf_beta,
        sf_gamma_factor=cfg.sf_gamma_factor,
        use_cautious=cfg.use_cautious,
    )
    autopilot = buselAutoPilot(
        opt_engine,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    return model, patcher, loss_engine, opt_engine, autopilot, total_params


def run(yaml_cfg, device, max_steps=5000):
    model, patcher, loss_engine, opt_engine, autopilot, total_params = build_model_and_opt(yaml_cfg, device)
    cfg = buselPretrainConfig.from_profile(yaml_cfg)

    d = yaml_cfg["model"]
    print(f"Model: {total_params:,} params ({total_params/1e6:.2f}M)")
    print(f"  d_model={d['d_model']}, layers={d['n_layers']}, heads={d['n_heads']}, experts={d['num_experts']}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"  Steps: {max_steps}, batch={yaml_cfg['data']['batch_size']}, chunk={yaml_cfg['data']['chunk_size']}")

    dataloader_iter = iter(get_busel_dataloader(
        data_path=yaml_cfg["data"]["data_path"],
        chunk_size=yaml_cfg["data"]["chunk_size"],
        batch_size=yaml_cfg["data"]["batch_size"],
    ))
    stride = patcher.stride

    csv_path = RESULTS_DIR / "data_capacity.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "tokens_per_sec", "tokens_seen", "wall_sec"])

    model.train()
    patcher.train()
    t_start = time.perf_counter()
    tokens_seen = 0
    best_loss = float("inf")
    best_step = 0

    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()

        byte_batch, _, _ = next(dataloader_iter)
        byte_batch = byte_batch.to(device, non_blocking=True)

        opt_engine.zero_grad(set_to_none=True)
        input_bytes = byte_batch[:, :-stride] if byte_batch.shape[1] > stride else byte_batch
        progress = step / max_steps

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            targets, mtp_targets = _build_targets(byte_batch, T_patches, stride=stride)
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(
                patches, [targets] + mtp_targets[:-1], progress=progress
            )
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets,
                [logits_t2, logits_t3, logits_t4],
                mtp_targets,
            ) + aux_loss.float()

        loss.backward()
        loss_val = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        autopilot.before_step(model, step, max_steps)
        autopilot.inject_noise(model)
        current_lr, _ = autopilot.update_parameters(step, loss_val, max_steps)
        opt_engine.step()

        if device == "cuda":
            torch.cuda.synchronize()

        step_wall = time.perf_counter() - step_t0
        tokens_this = byte_batch.shape[0] * byte_batch.shape[1]
        tokens_seen += tokens_this
        tok_s = tokens_this / step_wall if step_wall > 0 else 0
        total_wall = time.perf_counter() - t_start

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step

        if step % 100 == 0 or step == 1:
            csv_writer.writerow([step, f"{loss_val:.6f}", f"{tok_s:.0f}", tokens_seen, f"{total_wall:.2f}"])
            csv_file.flush()

            elapsed = total_wall
            rate = step / total_wall if total_wall > 0 else 0
            eta = (max_steps - step) / rate if rate > 0 else 0
            print(f"  step {step:>5d}/{max_steps}  loss={loss_val:.4f}  "
                  f"tok/s={tok_s:.0f}  tokens={tokens_seen/1e6:.0f}M  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    csv_file.close()

    final_loss = loss_val
    total_time = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Params:       {total_params:,}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Best loss:    {best_loss:.4f} (at step {best_step})")
    print(f"  Tokens seen:  {tokens_seen/1e9:.2f}B")
    print(f"  Total time:   {total_time:.0f}s")
    print(f"  CSV:          {csv_path}")
    print(f"{'='*60}")

    del model, patcher, loss_engine, opt_engine, autopilot
    torch.cuda.empty_cache()

    return csv_path


def plot(csv_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    rows = list(csv.DictReader(open(csv_path)))
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    tokens = [float(r["tokens_seen"]) / 1e6 for r in rows]
    times = [float(r["wall_sec"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Busel Data Capacity — How much data fits into 1.5M params?", fontsize=14, fontweight="bold")

    axes[0].plot(steps, losses, "o-", linewidth=2, markersize=4, color="#e74c3c")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss (nats)")
    axes[0].set_title("Loss vs Steps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tokens, losses, "s-", linewidth=2, markersize=4, color="#3498db")
    axes[1].set_xlabel("Tokens Seen (M)")
    axes[1].set_ylabel("Loss (nats)")
    axes[1].set_title("Loss vs Data")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, losses, "D-", linewidth=2, markersize=4, color="#2ecc71")
    axes[2].set_xlabel("Wall Time (seconds)")
    axes[2].set_ylabel("Loss (nats)")
    axes[2].set_title("Loss vs Time")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "data_capacity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    csv_path = RESULTS_DIR / "data_capacity.csv"

    if args.plot_only:
        if csv_path.exists():
            plot(csv_path)
        else:
            print(f"No CSV found at {csv_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Busel Data Capacity Test — device={device}")

    config_path = Path("configs/default.yaml")
    with open(config_path) as f:
        yaml_cfg = yaml.safe_load(f)["profiles"]["scale_m"]

    csv_path = run(yaml_cfg, device, max_steps=args.steps)
    plot(csv_path)


if __name__ == "__main__":
    main()

"""
🔬 Quick IMU-1 vs Baseline profiler — 100 steps on validation profile.
Compares: tok/s, peak VRAM, final loss, loss trajectory.

Usage:
    uv run python tests/quick_imu1_profile.py
    uv run python tests/quick_imu1_profile.py --steps 200
    uv run python tests/quick_imu1_profile.py --baseline-only
    uv run python tests/quick_imu1_profile.py --imu1-only
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine


def _build_model_and_engine(optimizer_type, lr_muon, lr_adamw, use_qknorm_l2=False,
                             lr_schedule="cosine", wsd_decay_fraction=0.2,
                             use_adafactor=False, device="cuda"):
    """Build model + optimizer engine for one config."""
    class Cfg:
        pass
    cfg = Cfg()
    cfg.vocab_size = _vocab_size()
    cfg.d_model = 128
    cfg.n_layers = 3
    cfg.n_heads = 4
    cfg.expert_hidden = 256
    cfg.num_experts = 2
    cfg.top_k = 1
    cfg.selective_backward = False
    cfg.backward_ratio = 1.0
    cfg.use_differential_attention = False
    cfg.use_qknorm_l2 = use_qknorm_l2
    cfg.n_hyper = 2

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    # bf16 norms
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__ and hasattr(module, "weight") and module.weight is not None:
                module.weight.data = module.weight.data.to(torch.bfloat16)

    if device == "cuda":
        model.enable_gradient_checkpointing(every=2)

    opt = buselOptimizerEngine(
        model,
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        optimizer_type=optimizer_type,
        lotus_rank=8,
        lotus_lr_scale=0.5,
        use_adafactor=use_adafactor,
    )
    autopilot = buselAutoPilot(
        opt,
        max_lr_muon=lr_muon,
        max_lr_adamw=lr_adamw,
        warmup_steps=10,
        min_lr_ratio=0.1,
        lr_schedule=lr_schedule,
        wsd_decay_fraction=wsd_decay_fraction,
    )
    loss_engine = buselLossEngine(vocab_size=cfg.vocab_size)
    return model, patcher, opt, autopilot, loss_engine, cfg


def _run_profile(name, model, patcher, opt, autopilot, loss_engine, dataloader,
                 steps, device, warmup=5):
    """Run training loop and collect metrics."""
    model.train()
    patcher.train()
    dl_iter = iter(dataloader)

    losses = []
    tok_per_s_list = []
    peak_mem_mb = 0.0

    def _step(model, patcher, opt, autopilot, loss_engine, dl_iter, device, step_num, max_steps):
        byte_batch, _, _ = next(dl_iter)
        byte_batch = byte_batch.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            targets = byte_batch[:, 1::patcher.stride][:, :T_patches]
            if targets.shape[1] < T_patches:
                targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)
            (logits_t1, _, _, _), aux_loss = model(patches, None)
            loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux_loss.float()
        loss.backward()
        autopilot.before_step(model, step_num, max_steps)
        opt.step()
        autopilot.update_parameters(step_num, loss.item(), max_steps)
        return loss.item(), byte_batch.shape[0] * byte_batch.shape[1]

    # Warmup
    for _ in range(warmup):
        _step(model, patcher, opt, autopilot, loss_engine, dl_iter, device, 0, steps)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Measured steps
    t0 = time.perf_counter()
    last_batch_tokens = 0
    for step in range(steps):
        loss_val, n_tokens = _step(model, patcher, opt, autopilot, loss_engine, dl_iter, device, step + warmup, steps + warmup)
        losses.append(loss_val)
        last_batch_tokens = n_tokens
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_per_s_list.append((step + 1) * n_tokens / elapsed)

    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.synchronize()

    elapsed_total = time.perf_counter() - t0
    total_tokens = steps * last_batch_tokens
    avg_tok_s = total_tokens / elapsed_total
    avg_loss = sum(losses) / len(losses) if losses else 0
    min_loss = min(losses) if losses else 0
    final_loss = losses[-1] if losses else 0

    return {
        "name": name,
        "steps": steps,
        "avg_tok_s": avg_tok_s,
        "avg_loss": avg_loss,
        "min_loss": min_loss,
        "final_loss": final_loss,
        "peak_mem_mb": peak_mem_mb,
        "elapsed_s": elapsed_total,
        "losses": losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick IMU-1 vs Baseline profiler")
    parser.add_argument("--steps", type=int, default=100, help="Measured steps (default: 100)")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--imu1-only", action="store_true", help="Run imu1 only")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"🔬 Quick IMU-1 vs Baseline profiler ({args.steps} steps, {device})")
    print("=" * 60)

    # Build dataloader
    dataloader = get_busel_dataloader(
        data_path="data_train",
        batch_size=16,
        chunk_size=256,
    )

    results = []

    # Baseline: lotus_muon, cosine, no QK-norm, no adafactor
    if not args.imu1_only:
        print("\n📊 BASELINE: lotus_muon + cosine LR")
        model_b, patcher_b, opt_b, ap_b, le_b, cfg_b = _build_model_and_engine(
            optimizer_type="lotus_muon",
            lr_muon=0.003,
            lr_adamw=0.0003,
            use_qknorm_l2=False,
            lr_schedule="cosine",
            use_adafactor=False,
            device=device,
        )
        r = _run_profile("baseline", model_b, patcher_b, opt_b, ap_b, le_b,
                          dataloader, args.steps, device)
        results.append(r)
        print(f"   tok/s: {r['avg_tok_s']:,.0f}  loss: {r['avg_loss']:.4f}  "
              f"min: {r['min_loss']:.4f}  final: {r['final_loss']:.4f}  "
              f"peak_mem: {r['peak_mem_mb']:.0f} MB  time: {r['elapsed_s']:.1f}s")
        del model_b, patcher_b, opt_b, ap_b
        if device == "cuda":
            torch.cuda.empty_cache()

    # IMU-1: norlotus_muon + WSD + QK-norm + adafactor
    if not args.baseline_only:
        print("\n📊 IMU-1: norlotus_muon + WSD + QK-Norm L2 + Adafactor")
        model_i, patcher_i, opt_i, ap_i, le_i, cfg_i = _build_model_and_engine(
            optimizer_type="norlotus_muon",
            lr_muon=0.003,
            lr_adamw=0.0003,
            use_qknorm_l2=True,
            lr_schedule="wsd",
            wsd_decay_fraction=0.2,
            use_adafactor=True,
            device=device,
        )
        r = _run_profile("imu1", model_i, patcher_i, opt_i, ap_i, le_i,
                          dataloader, args.steps, device)
        results.append(r)
        print(f"   tok/s: {r['avg_tok_s']:,.0f}  loss: {r['avg_loss']:.4f}  "
              f"min: {r['min_loss']:.4f}  final: {r['final_loss']:.4f}  "
              f"peak_mem: {r['peak_mem_mb']:.0f} MB  time: {r['elapsed_s']:.1f}s")
        del model_i, patcher_i, opt_i, ap_i
        if device == "cuda":
            torch.cuda.empty_cache()

    # Comparison
    if len(results) == 2:
        b, i = results
        print("\n" + "=" * 60)
        print("📊 COMPARISON")
        print("=" * 60)
        tok_s_delta = (i["avg_tok_s"] / b["avg_tok_s"] - 1) * 100
        loss_delta = (i["avg_loss"] / b["avg_loss"] - 1) * 100
        mem_delta = i["peak_mem_mb"] - b["peak_mem_mb"]
        print(f"  tok/s:     {b['avg_tok_s']:,.0f} → {i['avg_tok_s']:,.0f}  ({tok_s_delta:+.1f}%)")
        print(f"  avg loss:  {b['avg_loss']:.4f} → {i['avg_loss']:.4f}  ({loss_delta:+.1f}%)")
        print(f"  min loss:  {b['min_loss']:.4f} → {i['min_loss']:.4f}")
        print(f"  peak mem:  {b['peak_mem_mb']:.0f} MB → {i['peak_mem_mb']:.0f} MB  ({mem_delta:+.0f} MB)")
        print(f"  time:      {b['elapsed_s']:.1f}s → {i['elapsed_s']:.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()

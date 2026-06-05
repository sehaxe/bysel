"""
🧪 busel SHPAK 5-RUN PAIR-INTERACTION PROFILER — v5.8
Compares 5 configurations on shpak 52.8M to measure pair/triple interaction effects:
  1. baseline
  2. + LCSB ratio=0.5 (the proven winner from the per-feature ablation)
  3. + Sparse-BitNet 6:8 + LCSB
  4. + GradLite + LCSB
  5. + Sparse + GradLite + LCSB (all three)
Per run: 2 warmup + 10 measured steps, batch=16 ctx=4096, d_model=384 n_layers=8.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine


def load_shpak_profile() -> dict:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return full["profiles"]["shpak"]


def build_shpak_model_and_optim(cfg_profile, use_error_feedback=False, sparse_6_8=False,
                                  selective_backward=False, backward_ratio=1.0, device="cuda"):
    profile = dict(cfg_profile)
    profile["model"] = dict(profile["model"])
    profile["model"]["sparse_6_8"] = sparse_6_8
    profile["model"]["selective_backward"] = selective_backward
    profile["model"]["backward_ratio"] = backward_ratio
    profile["data"] = dict(profile["data"])
    profile["data"]["batch_size"] = 16
    profile["training"] = dict(profile["training"])
    profile["training"]["use_error_feedback"] = use_error_feedback

    class Cfg:
        pass
    cfg = Cfg()
    cfg.vocab_size = _vocab_size()
    cfg.optimizer_type = "lotus_muon"
    cfg.lotus_rank = 8
    cfg.lotus_lr_scale = 0.5
    m = profile["model"]
    for k, v in m.items():
        setattr(cfg, k, v)
    d = profile["data"]
    cfg.data_path = d["data_path"]
    cfg.chunk_size = d["chunk_size"]
    cfg.batch_size = d["batch_size"]
    t = profile["training"]
    for k, v in t.items():
        setattr(cfg, k, v)

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    target_dtype = torch.bfloat16
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)

    if device == "cuda":
        model.enable_gradient_checkpointing(every=2)

    opt = buselOptimizerEngine(
        model,
        lr_muon=cfg.learning_rate_muon,
        lr_adamw=cfg.learning_rate_adamw,
        optimizer_type=cfg.optimizer_type,
        lotus_rank=cfg.lotus_rank,
        lotus_lr_scale=cfg.lotus_lr_scale,
        use_error_feedback=use_error_feedback,
    )
    autopilot = buselAutoPilot(
        opt,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay,
    )
    loss_engine = buselLossEngine(cfg.vocab_size)
    return model, patcher, opt, autopilot, loss_engine, cfg


def run_one(name, cfg_profile, device="cuda", steps=10, **flags):
    print(f"\n{'=' * 80}\n🔬 RUN: {name}\n   flags: {flags}\n{'=' * 80}")
    model, patcher, opt, ap, loss_engine, cfg = build_shpak_model_and_optim(
        cfg_profile, device=device, **flags
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params: {n_params:,} ({n_params * 2 / 1024**2:.2f} MB FP16)")

    os.makedirs("data_train", exist_ok=True)
    test_file = "profiler_shpak_test_data.txt"
    created_dir = len(os.listdir("data_train")) == 0
    if created_dir:
        with open(os.path.join("data_train", test_file), "w", encoding="utf-8") as f:
            f.write("Шпак 5-ранов busel profiler. " * 300)
    try:
        dataloader = get_busel_dataloader(
            "data_train", chunk_size=cfg.chunk_size // 4, batch_size=cfg.batch_size
        )
        it = iter(dataloader)

        for _ in range(2):
            bb, _, _ = next(it)
            bb = bb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ib = bb[:, :-patcher.stride] if bb.shape[1] > patcher.stride else bb
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                patches = patcher(ib)
                T = patches.shape[1]
                tg = bb[:, 1::patcher.stride][:, :T]
                if tg.shape[1] < T:
                    tg = torch.nn.functional.pad(tg, (0, T - tg.shape[1]), value=0)
                (lo, _, _, _), aux = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(lo, tg) + aux.float()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        step_times, losses = [], []
        for s in range(steps):
            t0 = time.perf_counter()
            bb, _, _ = next(it)
            bb = bb.to(device, non_blocking=True)
            ib = bb[:, :-patcher.stride] if bb.shape[1] > patcher.stride else bb
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                patches = patcher(ib)
                T = patches.shape[1]
                tg = bb[:, 1::patcher.stride][:, :T]
                if tg.shape[1] < T:
                    tg = torch.nn.functional.pad(tg, (0, T - tg.shape[1]), value=0)
                (lo, _, _, _), aux = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(lo, tg) + aux.float()
            loss.backward()
            ap.inject_noise(model)
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            step_times.append(dt)
            losses.append(loss.item())

        mean_step = float(np.mean(step_times))
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        tokens_per_step = cfg.batch_size * cfg.chunk_size
        tps = tokens_per_step / mean_step
        print(f"   ✅ step={mean_step * 1000:.1f}ms  |  peak={peak_mb:.0f} MB  |  "
              f"tps={tps:.0f}  |  loss@10={np.mean(losses[-3:]):.3f}")
        return {
            "name": name,
            "step_ms": mean_step * 1000,
            "peak_mb": peak_mb,
            "tps": tps,
            "final_loss": float(np.mean(losses[-3:])),
        }
    finally:
        if created_dir:
            path = os.path.join("data_train", test_file)
            if os.path.exists(path):
                os.remove(path)
            try:
                os.rmdir("data_train")
            except OSError:
                pass


def main():
    print("🧪 busel SHPAK 5-RUN PAIR-INTERACTION PROFILER (v5.8)")
    print("   Profile: shpak 52.8M params, batch=16 ctx=4096")
    print("   Steps per run: 10 (2 warmup + 8 measured)\n")

    cfg_profile = load_shpak_profile()

    runs = [
        ("1. baseline", {}),
        ("2. + LCSB ratio=0.5", {"selective_backward": True, "backward_ratio": 0.5}),
        ("3. + Sparse 6:8 + LCSB", {
            "sparse_6_8": True, "selective_backward": True, "backward_ratio": 0.5,
        }),
        ("4. + GradLite + LCSB", {
            "use_error_feedback": True, "selective_backward": True, "backward_ratio": 0.5,
        }),
        ("5. + ALL three (Sparse+GradLite+LCSB)", {
            "sparse_6_8": True, "use_error_feedback": True,
            "selective_backward": True, "backward_ratio": 0.5,
        }),
    ]

    results = []
    for name, flags in runs:
        try:
            r = run_one(name, cfg_profile, **flags)
            results.append(r)
        except Exception as e:
            print(f"   ❌ FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": str(e)})

    print("\n" + "=" * 80)
    print("📊 SHPAK PAIR-INTERACTION COMPARISON (52.8M, batch=16 ctx=4096, 10 steps)".center(80))
    print("=" * 80)
    print(f"{'Run':<40} | {'Step (ms)':>10} | {'Peak (MB)':>10} | {'tok/s':>8} | {'Loss@10':>8}")
    print("-" * 95)
    base = results[0]
    for r in results:
        if "error" in r:
            print(f"{r['name']:<40} | {'ERR':>10} | {'ERR':>10} | {'ERR':>8} | {'ERR':>8}")
            continue
        ds = ""
        if r is not base and "step_ms" in base:
            delta_ms = r["step_ms"] - base["step_ms"]
            delta_mb = r["peak_mb"] - base["peak_mb"]
            ds = f"  (Δstep={delta_ms:+.1f}ms, Δmem={delta_mb:+.0f}MB)"
        print(f"{r['name']:<40} | {r['step_ms']:>10.1f} | {r['peak_mb']:>10.0f} | "
              f"{r['tps']:>8.0f} | {r['final_loss']:>8.3f}{ds}")
    print("=" * 95)


if __name__ == "__main__":
    main()

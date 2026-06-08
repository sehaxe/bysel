import argparse
import json
import math
import os
import sys
import time
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from model.layers import _BITLINEAR_CONFIG
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine

BATCH = 16
CHUNK_SIZE_FORCED = 4096
N_WARMUP = 2
N_MEASURE = 10
PROFILE = "shpak"


def _load_profile(name: str) -> dict:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return full["profiles"][name]


def _build(batch_size, optimizer_type="lotus_muon", lr_schedule="cosine",
           use_quest=False, quest_bits=1.58, wsd_s_enabled=False,
           wsd_s_interval=1000, wsd_s_decay_steps=200, device="cuda",
           use_tequila=False, tequila_lambda=1e-3,
           use_hestia=False, hestia_init_temp=6.0, hestia_end_temp=0.0):
    from model.layers import configure_bitlinear

    cfg_profile = _load_profile(PROFILE)
    profile = dict(cfg_profile)
    profile["model"] = dict(profile["model"])
    profile["model"]["selective_backward"] = True
    profile["model"]["backward_ratio"] = 0.5
    profile["data"] = dict(profile["data"])
    profile["data"]["batch_size"] = batch_size
    profile["data"]["chunk_size"] = CHUNK_SIZE_FORCED
    profile["training"] = dict(profile["training"])
    profile["training"]["optimizer_type"] = optimizer_type
    profile["training"]["lr_schedule"] = lr_schedule
    profile["training"]["use_quest"] = use_quest
    profile["training"]["quest_bits"] = quest_bits
    profile["training"]["wsd_s_enabled"] = wsd_s_enabled
    profile["training"]["wsd_s_interval"] = wsd_s_interval
    profile["training"]["wsd_s_decay_steps"] = wsd_s_decay_steps
    profile["training"]["use_tequila"] = use_tequila
    profile["training"]["tequila_lambda"] = tequila_lambda
    profile["model"]["use_hestia"] = use_hestia
    profile["training"]["hestia_init_temp"] = hestia_init_temp
    profile["training"]["hestia_end_temp"] = hestia_end_temp

    class Cfg:
        pass
    cfg = Cfg()
    cfg.vocab_size = _vocab_size()
    cfg.optimizer_type = optimizer_type
    cfg.lotus_rank = 8
    cfg.lotus_lr_scale = 0.5
    cfg.use_quest = use_quest
    cfg.quest_bits = quest_bits
    cfg.wsd_s_enabled = wsd_s_enabled
    cfg.wsd_s_interval = wsd_s_interval
    cfg.wsd_s_decay_steps = wsd_s_decay_steps
    cfg.lr_schedule = lr_schedule
    cfg.use_tequila = use_tequila
    cfg.tequila_lambda = tequila_lambda
    cfg.use_hestia = use_hestia
    cfg.hestia_init_temp = hestia_init_temp
    cfg.hestia_end_temp = hestia_end_temp
    for src in (profile["model"], profile["data"], profile["training"]):
        for k, v in src.items():
            setattr(cfg, k, v)

    # Configure global BitLinear flags BEFORE model construction
    hestia_temp = None
    if use_hestia:
        hestia_temp = torch.tensor(hestia_init_temp, device=device, dtype=torch.float32)
    configure_bitlinear(
        use_tequila=use_tequila,
        tequila_lambda=tequila_lambda,
        hestia_temperature=hestia_temp,
    )

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)
    target_dtype = torch.bfloat16
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__ and hasattr(module, "weight") and module.weight is not None:
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
        use_quest=cfg.use_quest,
        quest_bits=cfg.quest_bits,
    )
    autopilot = buselAutoPilot(opt, max_lr_muon=cfg.learning_rate_muon,
                               max_lr_adamw=cfg.learning_rate_adamw,
                               target_wd=cfg.weight_decay,
                               lr_schedule=cfg.lr_schedule,
                               wsd_s_enabled=cfg.wsd_s_enabled,
                               wsd_s_interval=cfg.wsd_s_interval,
                               wsd_s_decay_steps=cfg.wsd_s_decay_steps)
    loss_engine = buselLossEngine(cfg.vocab_size)
    return model, patcher, opt, autopilot, loss_engine, cfg


def _run_one(name, batch_size, device, n_warmup, n_measure, **kwargs):
    print(f"\n{'=' * 80}")
    print(f"🔬 RUN: {name}")
    print(f"   profile={PROFILE} batch={batch_size} device={device}")
    print(f"   kwargs={kwargs}")
    print(f"{'=' * 80}")

    model, patcher, opt, ap, loss_engine, cfg = _build(
        batch_size, device=device, **kwargs,
    )

    use_hestia = kwargs.get("use_hestia", False)
    hestia_init_temp = kwargs.get("hestia_init_temp", 6.0)
    hestia_end_temp = kwargs.get("hestia_end_temp", 0.0)
    total_steps = n_warmup + n_measure

    dataloader = iter(get_busel_dataloader(
        data_path=cfg.data_path,
        chunk_size=cfg.chunk_size // 4,
        batch_size=batch_size,
    ))

    step_times = []
    losses = []

    model.train()
    for step in range(total_steps):
        if use_hestia and device == "cuda":
            progress = step / max(total_steps - 1, 1)
            new_temp = hestia_end_temp + (hestia_init_temp - hestia_end_temp) * 0.5 * (1 + math.cos(math.pi * progress))
            _BITLINEAR_CONFIG["hestia_temperature"] = torch.tensor(new_temp, device=device, dtype=torch.float32)
            for m in model.modules():
                if hasattr(m, 'hestia_temperature') and m.hestia_temperature is not None:
                    m.hestia_temperature = torch.tensor(new_temp, device=device, dtype=torch.float32)

        bb, _, _ = next(dataloader)
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

        t0 = time.perf_counter()
        loss.backward()
        ap.before_step(model, step, total_steps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()
        ap.inject_noise(model)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

        temp_str = ""
        if use_hestia and step % 5 == 0:
            temp_str = f" temp={new_temp:.2f}"
        if step < n_warmup:
            print(f"  [warmup {step+1}/{n_warmup}] loss={loss.item():.4f}{temp_str}")
        else:
            print(f"  [step {step+1}/{total_steps}] loss={loss.item():.4f}{temp_str}")

    step_times = step_times[n_warmup:]
    losses = losses[n_warmup:]

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = batch_size * cfg.chunk_size
    mean_step = float(np.mean(step_times))
    std_step = float(np.std(step_times))
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
    tps = tokens_per_step / mean_step

    result = {
        "name": name,
        "batch_size": batch_size,
        "n_params": n_params,
        "step_ms": mean_step * 1000,
        "step_ms_std": std_step * 1000,
        "avg_loss": np.mean(losses),
        "min_loss": min(losses),
        "tps": tps,
        "peak_mb": peak_mb,
    }

    print(f"\n  📊 Results for {name}:")
    print(f"     params={n_params:,} ({n_params * 2 / 1024**2:.2f} MB FP16)")
    print(f"     avg_step={result['step_ms']:.1f}ms ± {result['step_ms_std']:.1f}ms")
    print(f"     tok/s={result['tps']:.0f}")
    print(f"     avg_loss={result['avg_loss']:.4f}, min_loss={result['min_loss']:.4f}")
    print(f"     peak_vram={result['peak_mb']:.0f}MB")

    return result


def main():
    parser = argparse.ArgumentParser(description="v6.2–v7.0 profiler on shpak 52.8M")
    parser.add_argument("--mode", choices=["all", "soap", "wsds", "wd33", "baseline",
                                            "tequila", "hestia", "muonq", "v7all"],
                        default="all", help="Which features to profile")
    parser.add_argument("--steps", type=int, default=20, help="Total steps per run (warmup=2, measure=steps-2)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device
    n_warmup = 2
    n_measure = args.steps - n_warmup

    results = []

    if args.mode in ("all", "baseline"):
        r = _run_one("baseline (lotus_muon + cosine)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="lotus_muon", lr_schedule="cosine")
        results.append(r)

    if args.mode in ("all", "soap"):
        r = _run_one("SOAP (Shampoo + Adam)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="soap")
        results.append(r)

    if args.mode in ("all", "wsds"):
        r = _run_one("WSD-S (checkpoint reuse)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="lotus_muon", lr_schedule="wsd",
                      wsd_s_enabled=True, wsd_s_interval=1000, wsd_s_decay_steps=200)
        results.append(r)

    if args.mode in ("all", "wd33"):
        r = _run_one("wd33 (warmdown-to-33%)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="lotus_muon", lr_schedule="wd33")
        results.append(r)

    if args.mode in ("v7all", "tequila"):
        r = _run_one("Tequila (deadzone reactivation)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="lotus_muon", lr_schedule="cosine",
                      use_tequila=True, tequila_lambda=1e-3)
        results.append(r)

    if args.mode in ("v7all", "hestia"):
        r = _run_one("Hestia (softmax relaxation)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="lotus_muon", lr_schedule="cosine",
                      use_hestia=True, hestia_init_temp=6.0, hestia_end_temp=0.0)
        results.append(r)

    if args.mode in ("v7all", "muonq"):
        r = _run_one("MuonQ (4-bit Muon)", BATCH, device, n_warmup, n_measure,
                      optimizer_type="muonq")
        results.append(r)

    print(f"\n{'=' * 80}")
    print("📊 SUMMARY — v6.2–v7.0 on shpak 52.8M")
    print(f"{'=' * 80}")
    print(f"{'Name':<40} {'tok/s':>8} {'step(ms)':>10} {'loss':>8} {'VRAM':>6}")
    print(f"{'-'*40} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")

    baseline_tps = results[0]["tps"] if results else 1
    baseline_step = results[0]["step_ms"] if results else 1

    for r in results:
        tok_delta = (r["tps"] / baseline_tps - 1) * 100
        step_delta = (r["step_ms"] / baseline_step - 1) * 100
        print(f"{r['name']:<40} {r['tps']:>7.0f} {r['step_ms']:>9.1f} {r['avg_loss']:>8.4f} {r['peak_mb']:>5.0f}MB")
        print(f"{'':>40} {'(' + f'{tok_delta:+.1f}%' + ')':>8} {'(' + f'{step_delta:+.1f}%' + ')':>10}")

    out_path = "checkpoints/v63_profile_results.json"
    os.makedirs("checkpoints", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    main()

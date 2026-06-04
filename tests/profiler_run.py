"""
📊 busel ULTRA-STABLE PROFILER v2.1
Оптимизирован для MPS (Apple Silicon) и CUDA.
Избегает использования нестабильного torch.profiler, вызывающего зависания на macOS.
Замеряет абсолютно все фазы шага обучения (Forward, Backward, Optimizer, Noise).

v2.1: --backend {auto, custom, torch} — auto uses torch.profiler на CUDA, custom на MPS/CPU.
"""

import os
import sys
import time
import resource
import torch
import torch.nn as nn

# Гарантируем корректность импортов из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine


class StablebuselTorchProfiler:
    """CUDA-only profiler using torch.profiler (kernel-level detail + Chrome trace export)."""

    def __init__(self, device="cuda", steps=10, trace_path="checkpoints/busel_profiler_trace.json"):
        self.device = device
        self.steps = steps
        self.trace_path = trace_path
        if device != "cuda":
            raise RuntimeError(f"StablebuselTorchProfiler requires CUDA (got {device}). torch.profiler hangs on MPS.")

    def get_memory_stats(self):
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def run_profiling(self, model, patcher, dataloader_iter, opt_engine, loss_engine, cfg):
        print(f"🔥 Warmup (2 steps)...", end=" ", flush=True)
        for _ in range(2):
            byte_batch, _, _ = next(dataloader_iter)
            byte_batch = byte_batch.to(self.device, non_blocking=True)
            opt_engine.zero_grad(set_to_none=True)
            input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                patches = patcher(input_bytes)
                T_patches = patches.shape[1]
                targets = byte_batch[:, 1::patcher.stride][:, :T_patches]
                if targets.shape[1] < T_patches:
                    targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)
                (logits_t1, _, _, _), aux_loss = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux_loss.float()
            loss.backward()
            opt_engine.step()
        torch.cuda.synchronize()
        print("✅")

        print(f"📊 torch.profiler collecting ({self.steps} steps)...")
        model.train()
        patcher.train()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for step in range(self.steps):
                byte_batch, _, _ = next(dataloader_iter)
                byte_batch = byte_batch.to(self.device, non_blocking=True)
                opt_engine.zero_grad(set_to_none=True)
                input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    patches = patcher(input_bytes)
                    T_patches = patches.shape[1]
                    targets = byte_batch[:, 1::patcher.stride][:, :T_patches]
                    if targets.shape[1] < T_patches:
                        targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)
                    (logits_t1, _, _, _), aux_loss = model(patches, None)
                    loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux_loss.float()
                loss.backward()
                opt_engine.step()
                prof.step()

        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        prof.export_chrome_trace(self.trace_path)
        return self.get_memory_stats(), prof

    def print_report(self, memory_stats, prof, total_params, cfg):
        print("\n" + "=" * 80)
        print("📊 busel TORCH.PROFILER REPORT (CUDA)".center(80))
        print("=" * 80)
        print(f"\n🧠 MODEL: {total_params:,} params ({total_params * 2 / 1024**2:.2f} MB FP16)")
        print(f"\n💾 MEMORY (CUDA):")
        for k, v in memory_stats.items():
            print(f"   • {k}: {v:.2f} MB")
        print("\n🔧 TOP 20 CUDA KERNELS (by total time):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"\n💾 Chrome trace saved to: {self.trace_path}")
        print("   Open in chrome://tracing or https://ui.perfetto.dev/ for full kernel timeline.")
        print("=" * 80)


class StablebuselProfiler:
    def __init__(self, device="mps", steps=10):
        self.device = device
        self.steps = steps

    def get_memory_stats(self):
        if self.device == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        elif self.device == "mps":
            return {
                "allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
            }
        
        usage = resource.getrusage(resource.RUSAGE_SELF)
        import platform
        if platform.system() == "Darwin":
            max_rss_mb = usage.ru_maxrss / (1024 * 1024)
        else:
            max_rss_mb = usage.ru_maxrss / 1024
        return {"max_rss_mb": max_rss_mb}

    def run_profiling(self, model, patcher, dataloader_iter, opt_engine, loss_engine, autopilot, cfg):
        print(f"🔥 Прогрев шейдеров и компиляция (warmup, 2 шага)...", end=" ", flush=True)
        # Разогрев
        for _ in range(2):
            byte_batch, _, _ = next(dataloader_iter)
            byte_batch = byte_batch.to(self.device, non_blocking=True)
            opt_engine.zero_grad(set_to_none=True)
            input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch

            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
                patches = patcher(input_bytes)
                T_patches = patches.shape[1]
                targets = byte_batch[:, 1::patcher.stride][:, :T_patches]
                if targets.shape[1] < T_patches:
                    targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)

                (logits_t1, _, _, _), aux_loss = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux_loss.float()

            loss.backward()
            opt_engine.step()
            
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        print("✅")

        # Начинаем детальный сбор статистики
        print(f"📊 Сбор детальной статистики ({self.steps} шагов)...")
        
        timings = {
            "data_loading": [],
            "patcher": [],
            "attention_moe_layers": [],
            "m_residuals": [],  # 🎯 ИСПРАВЛЕНО
            "final_norm": [],
            "backward_pass": [],
            "noise_injection": [],
            "optimizer_step": [],
            "total_step": []
        }
        
        model.train()
        patcher.train()
        
        for step in range(self.steps):
            step_start = time.perf_counter()
            
            # 1. Загрузка данных
            start = time.perf_counter()
            byte_batch, _, _ = next(dataloader_iter)
            byte_batch = byte_batch.to(self.device, non_blocking=True)
            if self.device == "mps": torch.mps.synchronize()
            timings["data_loading"].append(time.perf_counter() - start)
            
            opt_engine.zero_grad(set_to_none=True)
            input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
            
            # 2. Прямой проход - патчер
            start = time.perf_counter()
            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
                patches = patcher(input_bytes)
            if self.device == "mps": torch.mps.synchronize()
            timings["patcher"].append(time.perf_counter() - start)
            
            T_patches = patches.shape[1]
            targets = byte_batch[:, patcher.stride::patcher.stride][:, :T_patches]
            if targets.shape[1] < T_patches:
                targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)
            
            # 3. Декодерные слои (внутри бэкбона)
            layer_time = 0.0
            mres_time = 0.0
            x = patches
            n_hyper = getattr(model, "n_hyper", 2)
            streams = [x] * n_hyper  # pad with input copies (matches buselModel.forward)

            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
                for i, layer in enumerate(model.layers):
                    # Измеряем mAR (pre-mixing, как в buselModel.forward)
                    t0 = time.perf_counter()
                    mixed = model.m_residuals[i](x, streams)
                    if self.device == "mps": torch.mps.synchronize()
                    mres_time += (time.perf_counter() - t0)

                    # Измеряем декодерный слой
                    t0 = time.perf_counter()
                    x, aux = layer(mixed)
                    if self.device == "mps": torch.mps.synchronize()
                    layer_time += (time.perf_counter() - t0)

                    # Обновляем streams (FIFO shift, length stays = n_hyper)
                    streams = list(streams[1:]) + [x]
            
            timings["attention_moe_layers"].append(layer_time)
            timings["m_residuals"].append(mres_time)
            
            # 4. Финальная нормализация и лосс
            start = time.perf_counter()
            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
                hidden = model.final_norm(x)
                logits_t1 = model.mtp_pipeline.heads[0](hidden)
                loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux.float()
            if self.device == "mps": torch.mps.synchronize()
            timings["final_norm"].append(time.perf_counter() - start)
            
            # 5. Обратный проход (Backward)
            start = time.perf_counter()
            loss.backward()
            if self.device == "mps": torch.mps.synchronize()
            timings["backward_pass"].append(time.perf_counter() - start)
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. Впрыск шума Autopilot
            start = time.perf_counter()
            autopilot.inject_noise(model)
            if self.device == "mps": torch.mps.synchronize()
            timings["noise_injection"].append(time.perf_counter() - start)
            
            # 7. Шаг оптимизатора (Muon + AdamW)
            start = time.perf_counter()
            opt_engine.step()
            if self.device == "mps": torch.mps.synchronize()
            timings["optimizer_step"].append(time.perf_counter() - start)
            
            # Итоговый шаг
            timings["total_step"].append(time.perf_counter() - step_start)
            
        return timings

    def print_report(self, timings, memory_stats, total_params, cfg):
        import numpy as np
        
        print("\n" + "="*80)
        print("📊 busel FAST STABLE PROFILER REPORT".center(80))
        print("="*80)
        
        print(f"\n🧠 ПАРАМЕТРЫ МОДЕЛИ:")
        print(f"   • Всего параметров:  {total_params:,} ({total_params * 2 / 1024**2:.2f} MB в FP16)")
        
        print(f"\n💾 ПАМЯТЬ ({self.device.upper()}):")
        for k, v in memory_stats.items():
            print(f"   • {k}: {v:.2f} MB")
            
        avg_total = np.mean(timings["total_step"])
        
        print(f"\n⏱  ВРЕМЯ ШАГА (среднее по {self.steps} шагам):")
        print(f"   {'Компонент фазы шага':<40} | {'Среднее время':>15} | {'Доля %':>8}")
        print("   " + "-"*69)
        
        phases = [
            ("1. Загрузка данных (DataLoader CPU->GPU)", np.mean(timings["data_loading"])),
            ("2. FastBLTPatcher (Патчинг байтов)", np.mean(timings["patcher"])),
            ("3. Attention + MoE Layers (Все слои)", np.mean(timings["attention_moe_layers"])),
            ("4. mAR Residuals (Sinkhorn-связи)", np.mean(timings["m_residuals"])),
            ("5. Final Norm & Логиты (Конец Forward)", np.mean(timings["final_norm"])),
            ("6. Backward Pass (Расчет градиентов)", np.mean(timings["backward_pass"])),
            ("7. Autopilot Noise (Впрыск шума)", np.mean(timings["noise_injection"])),
            ("8. Optimizer Step (Muon + AdamW)", np.mean(timings["optimizer_step"])),
        ]
        
        for name, t in phases:
            pct = (t / avg_total) * 100
            print(f"   {name:<40} | {t*1000:12.2f} ms | {pct:6.1f}%")
            
        print("   " + "-"*69)
        print(f"   {'ИТОГО (Полный цикл обучения шага)':<40} | {avg_total*1000:12.2f} ms | {'100.0%':>8}")
        
        # Рекомендации
        print("\n" + "="*80)
        print("🎯 АНАЛИТИКА И РЕКОМЕНДАЦИИ".center(80))
        print("="*80)
        
        recs = []
        
        # Проверяем шум автопилота
        noise_time = np.mean(timings["noise_injection"])
        if noise_time > 0.05 * avg_total:
            pct = (noise_time / avg_total) * 100
            recs.append(
                f"🚨 Впрыск шума (inject_noise) занимает {pct:.1f}% времени шага ({noise_time*1000:.1f} ms).\n"
                f"      Рекомендация: Отключите inject_noise в train.py на Mac, так как randn_like на MPS крайне медленный."
            )
            
        # Проверяем оптимизатор
        opt_time = np.mean(timings["optimizer_step"])
        if opt_time > 0.15 * avg_total:
            pct = (opt_time / avg_total) * 100
            recs.append(
                f"⚠️  Шаг оптимизатора (Muon) занимает {pct:.1f}% времени шага ({opt_time*1000:.1f} ms).\n"
                f"      Рекомендация: Это нормально для некомпилированного Muon на CPU/MPS (выполняется 960 последовательных matmul).\n"
                f"      Для ускорения используйте видеокарты NVIDIA CUDA с компиляцией torch.compile."
            )
            
        # Проверяем mAR
        mar_time = np.mean(timings["m_residuals"])
        if mar_time > 0.15 * avg_total:
            pct = (mar_time / avg_total) * 100
            recs.append(
                f"⚠️  Блок Sinkhorn-связей mAR занимает {pct:.1f}% времени шага.\n"
                f"      Рекомендация: Сократите количество итераций Sinkhorn в mAR с 3 до 2."
            )
            
        if not recs:
            print("   ✅ Бутылочных горлышек не обнаружено! Архитектура работает на пределе возможностей железа.")
        else:
            for i, r in enumerate(recs, 1):
                print(f"   {i}. {r}\n")
                
        print(f"🚀 Пропускная способность (Throughput): {1/avg_total:.2f} шагов/сек")
        print(f"📈 Скорость обработки токенов: {cfg.batch_size * cfg.chunk_size / avg_total:.0f} токенов/сек")
        print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="busel Stable Step Profiler (v2.1)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "custom", "torch"],
                        help="Profiler backend: auto (torch on CUDA, custom on MPS/CPU), custom (manual perf_counter, stable everywhere), torch (torch.profiler, hangs on MPS — CUDA only).")
    parser.add_argument("--trace", type=str, default="checkpoints/busel_profiler_trace.json",
                        help="Output path for torch.profiler Chrome trace (only used with --backend torch).")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.backend == "auto":
        backend = "torch" if device == "cuda" else "custom"
    else:
        backend = args.backend

    if backend == "torch" and device != "cuda":
        print(f"⚠️  --backend torch is only safe on CUDA (current device: {device.upper()}). Falling back to custom.")
        backend = "custom"

    print(f"🖥️  Device: {device.upper()}  |  Profiler backend: {backend}")

    # Конфигурация для замера скорости (ziaziulia)
    class Config:
        vocab_size = _vocab_size()
        d_model = 256
        n_layers = 12
        n_heads = 4
        expert_hidden = 512
        num_experts = 4
        top_k = 2
        batch_size = 4
        chunk_size = 512
        data_path = "data_train"
        learning_rate_muon = 0.0004
        learning_rate_adamw = 0.00004
        weight_decay = 0.1
        
    cfg = Config()
    
    # Подготовка временных данных для замера
    test_file = "profiler_test_data.txt"
    created_dir = False
    if not os.path.exists(cfg.data_path) or len(os.listdir(cfg.data_path)) == 0:
        os.makedirs(cfg.data_path, exist_ok=True)
        created_dir = True
        with open(os.path.join(cfg.data_path, test_file), "w", encoding="utf-8") as f:
            f.write("Слава беларускаму аисту! Профайлер busel. Тестовые данные для замера. " * 300)
            
    # Загружаем DataLoader
    dataloader = get_busel_dataloader(cfg.data_path, chunk_size=cfg.chunk_size, batch_size=cfg.batch_size)
    dataloader_iter = iter(dataloader)
    
    # Инициализируем объекты
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)
    
    # Фикс типов для RMSNorm
    target_dtype = torch.bfloat16 if device == "cuda" else torch.float16
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)
                    
    opt_engine = buselOptimizerEngine(model, lr_muon=cfg.learning_rate_muon, lr_adamw=cfg.learning_rate_adamw)
    loss_engine = buselLossEngine(cfg.vocab_size)
    autopilot = buselAutoPilot(
        opt_engine,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay
    )
    
    # Запуск
    total_params = sum(p.numel() for p in model.parameters())

    try:
        if backend == "custom":
            profiler = StablebuselProfiler(device=device, steps=10)
            timings = profiler.run_profiling(model, patcher, dataloader_iter, opt_engine, loss_engine, autopilot, cfg)
            memory_stats = profiler.get_memory_stats()
            profiler.print_report(timings, memory_stats, total_params, cfg)
        elif backend == "torch":
            torch_profiler = StablebuselTorchProfiler(device=device, steps=10, trace_path=args.trace)
            memory_stats, prof = torch_profiler.run_profiling(model, patcher, dataloader_iter, opt_engine, loss_engine, cfg)
            torch_profiler.print_report(memory_stats, prof, total_params, cfg)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")
    finally:
        # Чистка
        if created_dir:
            path = os.path.join(cfg.data_path, test_file)
            if os.path.exists(path):
                os.remove(path)
            try:
                os.rmdir(cfg.data_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ busel TRAINING ENGINE v5.2 - Cybernetic Curriculum                         ║
║                                                                           ║
║ 🎯 KEY OPTIMIZATIONS:                                                     ║
║   • Sequence Length Warmup (Curriculum: 1024 -> 2048 -> 4096)             ║
║   • Dynamic MoE Router Scheduling (Adaptive aux_loss weighting)           ║
║   • Dynamic Chinchilla max_steps auto-calculator                          ║
║   • buselAutoPilot v6.0 (Predictive Gradient Dampening & Adaptive AGC)    ║
║   • CUDA-only Gradient Checkpointing (safeguards MPS RNG state bug)       ║
║   • Dynamic Auto-Batcher & Gradient Accumulation Integration              ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import signal
import argparse
import yaml
import json
import math

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from data.pipeline import get_busel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine, validate_training_schedule

from busel_logging import setup_logging, log_event
from ui import cli as ui
from ui.teto import frame as teto_frame
from ui.animation import teto_animate


class buselConfig:
    def __init__(self, profile_dict):
        self.d_model = profile_dict["model"]["d_model"]
        self.n_layers = profile_dict["model"]["n_layers"]
        self.n_heads = profile_dict["model"]["n_heads"]
        self.expert_hidden = profile_dict["model"]["expert_hidden"]
        self.num_experts = profile_dict["model"]["num_experts"]
        self.top_k = profile_dict["model"]["top_k"]
        self.vocab_size = profile_dict["model"]["vocab_size"]
        self.data_path = profile_dict["data"]["data_path"]
        self.chunk_size = profile_dict["data"]["chunk_size"]
        self.batch_size = profile_dict["data"]["batch_size"]
        self.weight_decay = profile_dict["training"]["weight_decay"]
        
        # Безопасно загружаем параметры планировщика с дефолтами
        self.max_steps = profile_dict["training"].get("max_steps", "auto")
        self.warmup_steps = profile_dict["training"].get("warmup_steps", "auto")
        self.min_lr_ratio = float(profile_dict["training"].get("min_lr_ratio", 0.1))

        # Накопление градиентов
        self.grad_accum_steps = int(profile_dict["training"].get("grad_accum_steps", 1))

        self.learning_rate_muon = profile_dict["training"].get("learning_rate_muon", 0.0006)
        self.learning_rate_adamw = profile_dict["training"].get("learning_rate_adamw", 0.00006)

        self.n_hyper = int(profile_dict["model"].get("n_hyper", 2))
        if self.d_model % self.n_hyper != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_hyper ({self.n_hyper})!")
        
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model dimension ({self.d_model}) must be divisible by n_heads ({self.n_heads})!")


def enforce_stability(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("   ⚙️  CUDA: TF32=ON, cuDNN.benchmark=ON")
    elif torch.backends.mps.is_available():
        print("   ⚙️  MPS: Metal Performance Shaders acceleration is active")


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _strip_compile_prefix(sd):
    """Strip torch.compile state_dict prefixes so checkpoints remain portable.

    `torch.compile(model, fullgraph=False)` wraps the model so that
    `model.state_dict()` returns keys prefixed with `_orig_mod.`. We strip this on
    resume so checkpoints saved by a compiled run can be loaded by an un-compiled
    resume (or vice-versa). Also handles `compiled_model.` and `_dynamo.`
    variants from older `torch._dynamo` versions.
    """
    if not sd:
        return sd
    out = {}
    for k, v in sd.items():
        new_k = k
        for prefix in ("_orig_mod.", "compiled_model.", "_dynamo."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        out[new_k] = v
    return out


def build_targets(byte_batch, input_length, stride=4):
    """
    ПРАВИЛЬНЫЕ таргеты для MTP-4:
    Патч i заканчивается на байте byte_{4i}
    T1 должен предсказывать byte_{4i+1} (следующий байт)
    T2 должен предсказывать byte_{4i+2}
    T3 должен предсказывать byte_{4i+3}
    T4 должен предсказывать byte_{4i+4}
    """
    # T1: следующий байт после патча
    targets = byte_batch[:, 1::stride][:, :input_length]
    if targets.shape[1] < input_length:
        pad_size = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad_size), value=0)
    
    # T2, T3, T4: следующие 3 байта
    mtp_targets = []
    for shift in [2, 3, 4]:  # ← ИСПРАВЛЕНО: было [1, 2, 3]
        mtp_target = byte_batch[:, shift::stride][:, :input_length]
        if mtp_target.shape[1] < input_length:
            pad_size = input_length - mtp_target.shape[1]
            mtp_target = torch.nn.functional.pad(mtp_target, (0, pad_size), value=0)
        mtp_targets.append(mtp_target)
    
    return targets, mtp_targets


def main():
    parser = argparse.ArgumentParser(description="busel v5.2 - Production Training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")
    parser.add_argument("--profile", type=str, default="shpak", help="Profile name from default.yaml")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default=balanced, reduce-overhead=+CUDA graphs, max-autotune=auto-tune kernels, slow compile)")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable gradient checkpointing")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  busel TRAINING ENGINE v5.2                                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    enforce_stability()
    busel_logger = setup_logging()
    log_event("training_start", profile=args.profile)

    with open("configs/default.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    if args.profile not in full_config["profiles"]:
        raise ValueError(f"Profile '{args.profile}' not found in configs/default.yaml")

    cfg = buselConfig(full_config["profiles"][args.profile])
    device = detect_device()

    ui.animated_header(
        f"busel TRAINING ENGINE v5.2 — profile: {args.profile}",
        subtitle=f"on {device.upper()} · 1-bit sovereign LLM",
        palette="teto",
        cycles=2,
    )
    ui.project_tree()
    print(f"\n🚀 Launching [busel-{args.profile}] on {device.upper()}")
    print(f"📚 Vocab: {cfg.vocab_size}, d_model: {cfg.d_model}, layers: {cfg.n_layers}")
    print(f"🧠 Experts: {cfg.num_experts}, Batch: {cfg.batch_size}, Target Context: {cfg.chunk_size}")
    if cfg.grad_accum_steps > 1:
        print(f"📦 Gradient Accumulation: ACTIVE ({cfg.grad_accum_steps} steps)")
    
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"Path '{cfg.data_path}' does not exist")

    start_step = 0
    start_file_idx = 0
    start_byte_offset = 0

    # === MODEL INITIALIZATION ===
    print("\n🔧 Initializing model...")
    with ui.spinner("Building model + patcher", style="dots"):
        patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
        model = buselModel(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params * 2 / 1024**2
    ui.status_panel(
        "MODEL",
        parameters=f"{total_params:,}",
        size_mb=f"{total_params_mb:.2f} MB",
        device=device.upper(),
    )
    log_event(
        "model_initialized",
        profile=args.profile,
        device=device,
        total_params=total_params,
        model_size_mb=round(total_params_mb, 2),
    )
    
    # === DYNAMIC MAX STEPS CALCULATION ===
    if cfg.max_steps == "auto" or cfg.max_steps is None:
        # Считаем шаги по константной глобальной нагрузке (Batch * grad_accum_steps * (Chunk // 4))
        global_batch_size = cfg.batch_size * cfg.grad_accum_steps
        tokens_per_step = global_batch_size * (cfg.chunk_size // 4)
        chinchilla_target_tokens = 80 * total_params
        cfg.max_steps = math.ceil(chinchilla_target_tokens / tokens_per_step)
        print(f"   📊 [CHINCHILLA AUTO-PLANNER] Activated:")
        print(f"      • Target Volume: {chinchilla_target_tokens:,} byte-tokens")
        print(f"      • Planned Steps: {cfg.max_steps:,}")
        log_event(
            "chinchilla_planned",
            target_tokens=chinchilla_target_tokens,
            planned_steps=cfg.max_steps,
            tokens_per_step=tokens_per_step,
            global_batch_size=global_batch_size,
        )
    else:
        cfg.max_steps = int(cfg.max_steps)

    if cfg.warmup_steps == "auto" or cfg.warmup_steps is None:
        cfg.warmup_steps = max(50, int(0.05 * cfg.max_steps))
    else:
        cfg.warmup_steps = int(cfg.warmup_steps)

    # Runtime guard (ISSUES.md #7): max_steps must exceed warmup_steps.
    cfg.max_steps, cfg.warmup_steps = validate_training_schedule(cfg.max_steps, cfg.warmup_steps)

    # Gradient Checkpointing (CUDA only)
    if device == "cuda" and not args.no_checkpointing:
        model.enable_gradient_checkpointing()

    _compile_in_progress = {"value": False}
    _emergency_save_requested = {"value": False}

    if device == "cuda" and not args.no_compile:
        compile_mode = args.compile_mode
        print(f"🔧 torch.compile (mode={compile_mode})...")
        _compile_in_progress["value"] = True
        try:
            model = torch.compile(model, fullgraph=False, dynamic=None, mode=compile_mode)
            patcher = torch.compile(patcher, fullgraph=False, dynamic=None, mode=compile_mode)
            print(f"   ✅ Compilation successful (mode={compile_mode})")
        except Exception as e:
            err_str = str(e)
            if "CUDAGraphs" in err_str or "FakeTensor" in err_str or "overwritten" in err_str:
                print(f"   ⚠️  {compile_mode} mode failed ({type(e).__name__}); falling back to default")
                try:
                    model = torch.compile(model, fullgraph=False, dynamic=None)
                    patcher = torch.compile(patcher, fullgraph=False, dynamic=None)
                    print("   ✅ Fallback to default-mode compile succeeded")
                except Exception as e2:
                    print(f"   ❌ Compile failed entirely: {type(e2).__name__}: {e2}")
            else:
                print(f"   ⚠️  Compile failed: {type(e).__name__}: {e}")
        finally:
            _compile_in_progress["value"] = False
            if _emergency_save_requested["value"]:
                print("💾 Running deferred emergency checkpoint save...")
                save_emergency_checkpoint(2, None)
    
    opt_engine = buselOptimizerEngine(model, lr_muon=cfg.learning_rate_muon, lr_adamw=cfg.learning_rate_adamw)
    autopilot = buselAutoPilot(
        opt_engine,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio
    )
    loss_engine = buselLossEngine(cfg.vocab_size)

    if args.resume and os.path.exists(args.resume):
        print(f"\n💾 Resuming checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(_strip_compile_prefix(checkpoint['model_state_dict']))
        patcher.load_state_dict(_strip_compile_prefix(checkpoint['patcher_state_dict']))

        if checkpoint.get('step') != 'emergency_backup':
            start_step = checkpoint['step']
            start_file_idx = checkpoint.get('file_idx', 0)
            start_byte_offset = checkpoint.get('byte_offset', 0)

    # === DATALOADER PREPARATION ===
    print("\n📚 Initializing Curriculum DataLoader...")
    current_chunk_size = cfg.chunk_size // 4
    current_batch_size = cfg.batch_size  # Динамический батч
    
    dataloader = get_busel_dataloader(
        cfg.data_path, 
        chunk_size=current_chunk_size, 
        batch_size=current_batch_size,
        start_file_idx=start_file_idx,
        start_byte_offset=start_byte_offset
    )

    global_current_file_idx = start_file_idx
    global_current_byte_offset = start_byte_offset
    global_current_step = start_step

    def save_emergency_checkpoint(signum, frame):
        if _compile_in_progress["value"] or _emergency_save_requested["value"]:
            return
        _emergency_save_requested["value"] = True
        print("\n\n💾 [SIGINT] Emergency save requested — will save at next safe step boundary.")
        try:
            log_event(
                "emergency_save_requested",
                step=global_current_step,
                signal=signum,
            )
        except Exception:
            pass

    signal.signal(signal.SIGINT, save_emergency_checkpoint)
    signal.signal(signal.SIGTERM, save_emergency_checkpoint)

    # Автокаст на CPU не поддерживает float16, поэтому разделяем типы
    if device == "cuda" or device == "cpu":
        autocast_dtype = torch.bfloat16
    else:  # mps
        autocast_dtype = torch.float16
        
    autocast_enabled = (device in ["cuda", "mps"])
    
    use_cuda_stream = (device == "cuda")
    prefetch_stream = None
    dataloader_iter = iter(dataloader)
    current_batch = None
    
    if use_cuda_stream:
        prefetch_stream = torch.cuda.Stream()
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            return
    else:
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            return

    print("\n🔥 Training started.")
    print("=" * 100)
    
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0
    
    # Кумулятивный счетчик обработанных токенов
    cumulative_processed_tokens = start_step * current_batch_size * cfg.grad_accum_steps * current_chunk_size
    
    for step_offset in range(cfg.max_steps):
        step = start_step + step_offset
        global_current_step = step

        progress = float(step) / float(cfg.max_steps)

        # 🎯 ДИНАМИЧЕСКИЙ CURRICULUM ДЛИНЫ КОНТЕКСТА:
        new_chunk_size = current_chunk_size
        if progress < 0.15:
            new_chunk_size = cfg.chunk_size // 4
        elif progress < 0.35:
            new_chunk_size = cfg.chunk_size // 2
        else:
            new_chunk_size = cfg.chunk_size
            
        # Если пришел момент смены фазы — переинициализируем DataLoader, сохраняя позицию в датасете
        if new_chunk_size != current_chunk_size:
            # 🎯 DYNAMIC AUTO-BATCHER:
            # Вычисляем новый батч обратно пропорционально размеру нового контекста,
            # чтобы удержать общее потребление памяти VRAM идеально стабильным.
            new_batch_size = max(1, (cfg.batch_size * (cfg.chunk_size // 4)) // new_chunk_size)
            
            print(f"\n📈 [CURRICULUM UPGRADE]: Progress {progress*100:.1f}% -> Scaling context window from {current_chunk_size} to {new_chunk_size} & Auto-adapting Batch from {current_batch_size} to {new_batch_size}!")
            log_event(
                "curriculum_upgrade",
                step=step,
                progress=round(progress, 4),
                old_chunk=current_chunk_size,
                new_chunk=new_chunk_size,
                old_batch=current_batch_size,
                new_batch=new_batch_size,
            )
            
            current_chunk_size = new_chunk_size
            current_batch_size = new_batch_size
            
            dataloader = get_busel_dataloader(
                cfg.data_path, 
                chunk_size=current_chunk_size, 
                batch_size=current_batch_size,
                start_file_idx=global_current_file_idx,
                start_byte_offset=global_current_byte_offset
            )
            dataloader_iter = iter(dataloader)
            try:
                current_batch = next(dataloader_iter)
            except StopIteration:
                print("📝 Dataset ended during Curriculum switch.")
                break
        
        # Очищаем градиенты перед циклом накопления для текущего шага оптимизатора
        opt_engine.zero_grad(set_to_none=True)
        
        accumulated_loss = 0.0
        accumulated_aux_loss = 0.0
        
        # Внутренний цикл по шагам накопления градиентов (Gradient Accumulation)
        for accum_step in range(cfg.grad_accum_steps):
            if current_batch is None:
                break
                
            byte_batch, last_file_idx, last_byte_offset = current_batch
            byte_batch = byte_batch.to(device, non_blocking=True)
            
            global_current_file_idx = last_file_idx
            global_current_byte_offset = last_byte_offset
            
            input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
            
            # Step 2. Forward pass (передаем текущий прогресс для MoE-роутера)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
                patches = patcher(input_bytes)
                T_patches = patches.shape[1]
                
                targets, mtp_targets = build_targets(
                    byte_batch, T_patches, stride=patcher.stride
                )
                
                (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(patches, [targets] + mtp_targets[:-1], progress=progress)
                
                loss = loss_engine.compute_pretrain_loss(
                    logits_t1, targets,
                    [logits_t2, logits_t3, logits_t4],
                    mtp_targets
                ) + aux_loss.float()
            
            # Делим лосс на число шагов накопления
            loss = loss / cfg.grad_accum_steps
            
            # Step 3. Backward pass (аккумулирует градиенты)
            loss.backward()
            
            accumulated_loss += loss.item() * cfg.grad_accum_steps
            accumulated_aux_loss += aux_loss.item()
            
            # Подгружаем следующий батч асинхронно
            next_batch = None
            if use_cuda_stream:
                with torch.cuda.stream(prefetch_stream):
                    try:
                        next_batch = next(dataloader_iter)
                    except StopIteration:
                        next_batch = None
            else:
                try:
                    next_batch = next(dataloader_iter)
                except StopIteration:
                    next_batch = None
            
            if use_cuda_stream:
                torch.cuda.current_stream().wait_stream(prefetch_stream)
                
            # Учитываем объем обработанных токенов на этом микро-шаге
            tokens_this_step = current_batch_size * current_chunk_size
            cumulative_processed_tokens += tokens_this_step
            
            current_batch = next_batch
            
        if current_batch is None and accumulated_loss == 0.0:
            print("\n🎉 Dataset completely processed.")
            break
            
        # Адаптивный клиппинг и превентивное подавление взрывов по накопленным градиентам
        dynamic_clip = autopilot.before_step(model, step, cfg.max_steps)
        
        if device == "cuda":
            autopilot.inject_noise(model)
            
        current_lr, noise_scale = autopilot.update_parameters(step, accumulated_loss, cfg.max_steps)

        # Обновляем веса один раз за шаг оптимизатора
        opt_engine.step()

        # 🎯 SAFE-POINT: Handle deferred emergency save (set by SIGINT handler).
        # At this point we are past forward/backward/step and not inside torch.compile
        # tracing, so calling model.state_dict() is safe (returns real tensors).
        if _emergency_save_requested["value"]:
            try:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'model_state_dict': _strip_compile_prefix(model.state_dict()),
                    'patcher_state_dict': _strip_compile_prefix(patcher.state_dict()),
                    'step': global_current_step,
                    'file_idx': global_current_file_idx,
                    'byte_offset': global_current_byte_offset,
                }, "checkpoints/latest_crash_backup.pt")
                log_event(
                    "emergency_checkpoint",
                    step=global_current_step,
                    path="checkpoints/latest_crash_backup.pt",
                )
                print("💾 Emergency checkpoint saved (deferred).")
            except Exception as save_err:
                print(f"❌ Emergency save failed: {type(save_err).__name__}: {save_err}")
            finally:
                _emergency_save_requested["value"] = False
            print("Exiting after emergency save.")
            sys.exit(0)
        
        # Step 4. Fast interval logging
        if step % 10 == 0:
            current_time = time.time()
            
            if step_offset == 0:
                elapsed_interval = current_time - start_time
                tokens_interval = tokens_this_step * cfg.grad_accum_steps
            else:
                elapsed_interval = current_time - last_log_time
                tokens_interval = cumulative_processed_tokens - last_log_tokens
            
            speed = tokens_interval / elapsed_interval if elapsed_interval > 0 else 0
            
            last_log_time = current_time
            last_log_tokens = cumulative_processed_tokens
            
            vram = ""
            vram_mb = 0.0
            if device == "cuda":
                vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                vram = f" | VRAM: {vram_mb:.0f}MB"
            elif device == "mps":
                vram_mb = torch.mps.current_allocated_memory() / 1024**2
                vram = f" | VRAM: {vram_mb:.0f}MB"
            
            print(
                f"Step {step:05d}/{cfg.max_steps:05d} | "
                f"Total: {accumulated_loss:.2f} | "
                f"Aux: {accumulated_aux_loss / cfg.grad_accum_steps:.2f} | "
                f"LR: {current_lr:.5f} | "
                f"Clip: {dynamic_clip:.2f} | "
                f"Batch: {current_batch_size} | "
                f"{speed:.0f} tokens/s{vram}"
            )

            metrics = {
                "step": step,
                "loss": accumulated_loss,
                "aux_loss": accumulated_loss / cfg.grad_accum_steps,
                "lr": current_lr,
                "speed": speed,
                "vram": vram_mb
            }
            os.makedirs("checkpoints", exist_ok=True)
            with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as log_f:
                log_f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            log_event(
                "step_complete",
                step=step,
                loss=round(accumulated_loss, 4),
                aux_loss=round(accumulated_aux_loss / cfg.grad_accum_steps, 4),
                lr=round(current_lr, 7),
                tokens_per_s=round(speed, 1),
                vram_mb=round(vram_mb, 1),
                batch=current_batch_size,
                chunk=current_chunk_size,
            )

        # Step 5. Scheduled checkpoint (каждые 100 шагов)
        if step % 100 == 0 and step > 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/busel_{args.profile}_step_{step}.pt"
            temp_path = checkpoint_path + ".tmp"
            
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'patcher_state_dict': patcher.state_dict(),
                    'step': step,
                    'file_idx': last_file_idx,
                    'byte_offset': last_byte_offset,
                    'loss': accumulated_loss,
                    'lr_muon': current_lr,
                    'profile': args.profile,
                }, temp_path)
                
                file_size = os.path.getsize(temp_path)
                expected_min = 2_000_000  # 2 MB minimum for any model
                if file_size < expected_min:
                    print(f"⚠️  WARNING: Checkpoint too small ({file_size / 1024:.1f} KB). Possible corruption!")
                    os.remove(temp_path)
                    log_event("checkpoint_rejected", step=step, file_size=file_size, reason="too_small")
                else:
                    os.rename(temp_path, checkpoint_path)
                    print(f"💾 Scheduled checkpoint saved: {checkpoint_path} ({file_size / 1024 / 1024:.1f} MB)")
                    log_event(
                        "checkpoint_saved",
                        step=step,
                        path=checkpoint_path,
                        file_size_mb=round(file_size / 1024 / 1024, 2),
                        loss=round(accumulated_loss, 4),
                        lr=round(current_lr, 7),
                    )
            except Exception as e:
                print(f"❌ Failed to save checkpoint: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                import logging as _logging
                log_event("checkpoint_failed", step=step, error=str(e), level=_logging.WARNING)
    # === FINAL ===
    total_time = time.time() - start_time
    avg_speed = cumulative_processed_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 100)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"   Total time:   {total_time/3600:.2f} h")
    print(f"   Total tokens: {cumulative_processed_tokens:,}")
    print(f"   Avg speed:    {avg_speed:.1f} tokens/sec")

    os.makedirs("checkpoints", exist_ok=True)
    final_path = f"checkpoints/busel_{args.profile}_FINAL.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'patcher_state_dict': patcher.state_dict(),
        'step': global_current_step,
        'file_idx': global_current_file_idx,
        'byte_offset': global_current_byte_offset,
        'profile': args.profile,
        'config': vars(cfg),
    }, final_path)
    print(f"💾 Final checkpoint: {final_path}")
    log_event(
        "training_complete",
        profile=args.profile,
        total_steps=global_current_step,
        total_time_s=round(total_time, 1),
        total_tokens=cumulative_processed_tokens,
        avg_tokens_per_s=round(avg_speed, 1),
        final_path=final_path,
    )


if __name__ == "__main__":
    main()
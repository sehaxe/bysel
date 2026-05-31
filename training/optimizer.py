"""
⚙️ BYSEL OPTIMIZER - Official Muon Specification (FP32 Newton-Schulz) & AdamW
"""
import torch
import math
import platform

try:
    from flash_muon import Muon as FlashMuon
    HAS_FLASH_MUON = True
except ImportError:
    HAS_FLASH_MUON = False

def _newton_schulz_core(X, steps=5):
    """Ядро итераций Ньютона-Шульца без мертвого кода."""
    original_dtype = X.dtype
    X = X.float()
    X = X / (X.norm() + 1e-8)
    a1, b1, c1 = 3.4445, -4.7750, 2.0315
    is_tall = X.size(0) > X.size(1)
    if is_tall:
        X = X.transpose(0, 1)
        
    for _ in range(steps):
        XXT = torch.matmul(X, X.transpose(-1, -2))
        # 🎯 ИСПРАВЛЕНИЕ: Убран недостижимый if/else (steps=5 < 8)
        X = a1 * X + b1 * torch.matmul(XXT, X) + c1 * torch.matmul(torch.matmul(XXT, XXT), X)
        
    if is_tall:
        X = X.transpose(0, 1)
    return X.to(original_dtype)

if platform.system() == "Linux" and torch.cuda.is_available():
    try:
        @torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
        def _compiled_newton_schulz(X, steps=5):
            return _newton_schulz_core(X, steps)
    except Exception:
        def _compiled_newton_schulz(X, steps=5):
            return _newton_schulz_core(X, steps)
else:
    def _compiled_newton_schulz(X, steps=5):
        return _newton_schulz_core(X, steps)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    if p.device.type == "cuda": dtype = torch.bfloat16
                    elif p.device.type == "mps": dtype = torch.float16
                    else: dtype = torch.float32
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=dtype)
                    
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad.to(buf.dtype))
                m_t = grad.to(buf.dtype) + momentum * buf
                
                O_t = _compiled_newton_schulz(m_t, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
                
                p.mul_(1.0 - lr * wd)
                p.add_(O_t.to(p.dtype), alpha=-lr * scale)

    def hybrid_newton_schulz(self, M, steps=10):
        return _newton_schulz_core(M, steps)

class ByselOptimizerEngine:
    def __init__(self, model, lr_muon=0.002, lr_adamw=0.0002):
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim == 2 and "router" not in name and "proj" in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        if len(muon_params) > 0:
            if HAS_FLASH_MUON and torch.cuda.is_available():
                print("🚀 [CUDA ULTRA-SPEED]: Активирован Triton-оптимизатор Flash-Muon!")
                self.opt_muon = FlashMuon(muon_params, lr=lr_muon, momentum=0.95)
            else:
                self.opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)
        else:
            self.opt_muon = None
            adamw_params.extend(muon_params)
            
        self.opt_adamw = torch.optim.AdamW(adamw_params, lr=lr_adamw, weight_decay=0.01)

    def zero_grad(self, set_to_none: bool = True):
        if self.opt_muon is not None: self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.opt_muon is not None: self.opt_muon.step()
        self.opt_adamw.step()
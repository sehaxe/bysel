"""
⚙️ busel OPTIMIZER - Official Muon Specification (FP32 Newton-Schulz) & AdamW
"""
import torch
import math
import platform
from busel_registry import register

try:
    from flash_muon import Muon as FlashMuon
    HAS_FLASH_MUON = True
except ImportError:
    HAS_FLASH_MUON = False

def _newton_schulz_core(X, steps=5):
    """Newton-Schulz quintic iteration. Uses Frobenius norm (Keller Jordan spec).

    NOTE (ISSUES.md #4): Earlier attempt to use spectral norm initial
    normalisation diverged — spectral norm <= Frobenius, so dividing by it
    scales the matrix UP into the divergent region. Frobenius is correct.
    """
    original_dtype = X.dtype
    X = X.float()
    X = X / (X.norm() + 1e-8)

    a1, b1, c1 = 3.4445, -4.7750, 2.0315
    is_tall = X.size(0) > X.size(1)
    if is_tall:
        X = X.transpose(0, 1)

    for _ in range(steps):
        XXT = torch.matmul(X, X.transpose(-1, -2))
        X = a1 * X + b1 * torch.matmul(XXT, X) + c1 * torch.matmul(torch.matmul(XXT, XXT), X)

    if is_tall:
        X = X.transpose(0, 1)
    return X.to(original_dtype)

if platform.system() == "Linux" and torch.cuda.is_available():
    try:
        @torch.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
        def _compiled_newton_schulz(X, steps=5):
            return _newton_schulz_core(X, steps)
    except Exception:
        def _compiled_newton_schulz(X, steps=5):
            return _newton_schulz_core(X, steps)
else:
    def _compiled_newton_schulz(X, steps=5):
        return _newton_schulz_core(X, steps)

@register("optimizer", "muon")
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
                # Keller Jordan Muon spec: m_t = buf (post-update). Fixes ISSUES.md #3.
                m_t = buf

                O_t = _compiled_newton_schulz(m_t, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))

                p.mul_(1.0 - lr * wd)
                p.add_(O_t.to(p.dtype), alpha=-lr * scale)

    def hybrid_newton_schulz(self, M, steps=10):
        return _newton_schulz_core(M, steps)

@register("optimizer", "lotus_muon")
class LotusMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 ns_steps=5, rank=8, lr_scale=0.5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, rank=rank, lr_scale=lr_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            rank = group['rank']
            lr_scale = group['lr_scale']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                d1, d2 = p.shape
                if 'buf_p' not in state:
                    eff_rank = min(rank, d1, d2)
                    if p.device.type == "cuda": dtype = torch.bfloat16
                    elif p.device.type == "mps": dtype = torch.float16
                    else: dtype = torch.float32
                    state['buf_p'] = torch.zeros(d1, eff_rank, dtype=dtype, device=p.device)
                    state['buf_q'] = torch.zeros(d2, eff_rank, dtype=dtype, device=p.device)

                buf_p = state['buf_p']
                buf_q = state['buf_q']
                buf_p.mul_(momentum).add_(grad.to(buf_p.dtype) @ buf_q)
                buf_q.mul_(momentum).add_(grad.to(buf_q.dtype).T @ buf_p)

                m_approx = buf_p @ buf_q.T
                O_t = _compiled_newton_schulz(m_approx, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))

                p.mul_(1.0 - lr * wd)
                p.add_(O_t.to(p.dtype), alpha=-lr * scale * lr_scale)

@register("optimizer", "hybrid_muon_adamw")
class buselOptimizerEngine:
    """Hybrid Muon (2D weight params, no router/embed) + AdamW (1D/norm/bias/embed/router)."""

    _MUON_EXCLUDE = ("router", "embed")
    _LR_GROUPS = ("attn", "ffn", "mtp", "norm", "embed", "router")

    @staticmethod
    def _classify_param(name: str) -> str:
        n = name.lower()
        if "router" in n: return "router"
        if "embed" in n: return "embed"
        if "norm" in n: return "norm"
        if "mtp" in n: return "mtp"
        if "ffn" in n or "blackboard" in n: return "ffn"
        if any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "wk", "wv", "wq")):
            return "attn"
        if "moe" in n: return "ffn"
        return "attn"

    def __init__(self, model, lr_muon=0.002, lr_adamw=0.0002,
                 optimizer_type="muon", lotus_rank=8, lotus_lr_scale=0.5,
                 lr_multipliers=None):
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_muon_param = (
                param.ndim == 2
                and all(token not in name for token in self._MUON_EXCLUDE)
            )
            if is_muon_param:
                muon_params.append((name, param))
            else:
                adamw_params.append((name, param))

        mults = dict(lr_multipliers or {})
        for k in self._LR_GROUPS:
            mults.setdefault(k, 1.0)
        self._lr_mults = mults

        def _build_subgroups(items):
            groups: dict[str, list] = {k: [] for k in self._LR_GROUPS}
            for name, p in items:
                groups[self._classify_param(name)].append(p)
            out = []
            for k, params in groups.items():
                if params:
                    out.append({"params": params, "lr_mult": mults[k], "name": k})
            return out

        if len(muon_params) > 0:
            muon_groups = _build_subgroups(muon_params)
            if optimizer_type == "lotus_muon":
                print(f"🪷 [LOTUS-MUON]: rank={lotus_rank}, lr_scale={lotus_lr_scale} — "
                      f"~{10.0 / max(1, lotus_rank):.1f}x optimizer-state memory reduction")
                self.opt_muon = LotusMuon(
                    muon_groups, lr=lr_muon, momentum=0.95,
                    rank=lotus_rank, lr_scale=lotus_lr_scale,
                )
            elif HAS_FLASH_MUON and torch.cuda.is_available():
                print("🚀 [CUDA ULTRA-SPEED]: Активирован Triton-оптимизатор Flash-Muon!")
                self.opt_muon = FlashMuon(muon_groups, lr=lr_muon, momentum=0.95)
            else:
                self.opt_muon = Muon(muon_groups, lr=lr_muon, momentum=0.95)
        else:
            self.opt_muon = None

        if self.opt_muon is None:
            adamw_groups = _build_subgroups(muon_params + adamw_params)
        else:
            adamw_groups = _build_subgroups(adamw_params)

        self.opt_adamw = torch.optim.AdamW(adamw_groups, lr=lr_adamw, weight_decay=0.01)
        self.optimizer_type = optimizer_type

        n_muon = sum(p.numel() for _, p in muon_params)
        n_adamw = sum(p.numel() for _, p in adamw_params)
        n_total = n_muon + n_adamw
        if n_total > 0:
            mults_str = " | ".join(f"{k}={v:.2f}" for k, v in mults.items() if v != 1.0)
            suffix = f"  ⚖️  LR mults: {mults_str}" if mults_str else ""
            print(f"⚙️  Hybrid optimiser routing: {n_muon:,} → {optimizer_type} "
                  f"({100.0 * n_muon / n_total:.1f}%), {n_adamw:,} → AdamW "
                  f"({100.0 * n_adamw / n_total:.1f}%){suffix}")

    def zero_grad(self, set_to_none: bool = True):
        if self.opt_muon is not None: self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.opt_muon is not None: self.opt_muon.step()
        self.opt_adamw.step()


@register("optimizer", "ema")
class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self._step_count = 0
        self.shadow = {
            k: v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self._step_count += 1
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k].copy_(self.shadow[k].to(sd[k].dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup: dict):
        sd = model.state_dict()
        for k in sd:
            sd[k].copy_(backup[k].to(sd[k].dtype))

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict):
        self.shadow = {
            k: (v.float() if v.dtype.is_floating_point else v.clone())
            for k, v in sd.items()
        }
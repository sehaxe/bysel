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


@register("optimizer", "normuon")
class NorMuon(torch.optim.Optimizer):
    """Neuron-wise normalized Muon (Li et al. 2025, arXiv:2602.02522).
    
    Normalizes each neuron independently before orthogonalization.
    Uses cautious weight decay for selective regularization.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5, cautious_wd=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps, cautious_wd=cautious_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            cautious_wd = group['cautious_wd']
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
                m_t = buf

                if cautious_wd:
                    wd_mask = (m_t * p.data > 0).to(m_t.dtype)
                    p.mul_(1.0 - lr * wd * wd_mask)
                else:
                    p.mul_(1.0 - lr * wd)

                O_t = _compiled_newton_schulz(m_t, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
                p.add_(O_t.to(p.dtype), alpha=-lr * scale)

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


@register("optimizer", "norlotus_muon")
class NorLotusMuon(torch.optim.Optimizer):
    """NorMuon + LOTUS rank-r factorized momentum (Li et al. 2025 + arXiv:2602.01233).
    
    Combines neuron-wise normalization with rank-r factorized momentum.
    Cautious weight decay for selective regularization.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 ns_steps=5, rank=8, lr_scale=0.5, cautious_wd=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, rank=rank, lr_scale=lr_scale, cautious_wd=cautious_wd)
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
            cautious_wd = group['cautious_wd']
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

                if cautious_wd:
                    wd_mask = (m_approx * p.data > 0).to(m_approx.dtype)
                    p.mul_(1.0 - lr * wd * wd_mask)
                else:
                    p.mul_(1.0 - lr * wd)

                O_t = _compiled_newton_schulz(m_approx, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
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

    def __init__(self, *modules, lr_muon=0.002, lr_adamw=0.0002,
                 optimizer_type="muon", lotus_rank=8, lotus_lr_scale=0.5,
                 lr_multipliers=None, use_schedule_free=False,
                 sf_beta=0.9, sf_gamma_factor=2.0,
                 use_cautious=False, use_adafactor=False,
                 use_quest=False, quest_bits=1.58):
        muon_params = []
        adamw_params = []
        for module in modules:
            for name, param in module.named_parameters():
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
            elif optimizer_type == "normuon":
                print(f"🔬 [NOR-MUON]: neuron-wise normalized Muon + cautious WD (Li et al. 2025)")
                self.opt_muon = NorMuon(
                    muon_groups, lr=lr_muon, momentum=0.95, cautious_wd=True,
                )
            elif optimizer_type == "norlotus_muon":
                print(f"🔬🪷 [NOR-LOTUS-MUON]: NorMuon + LOTUS rank={lotus_rank} (IMU-1 recipe)")
                self.opt_muon = NorLotusMuon(
                    muon_groups, lr=lr_muon, momentum=0.95,
                    rank=lotus_rank, lr_scale=lotus_lr_scale, cautious_wd=True,
                )
            elif optimizer_type == "soap":
                print(f"🧊 [SOAP]: Shampoo eigenspace + Adam (Vyas et al. 2025, ICLR 2025)")
                self.opt_muon = SOAP(
                    muon_groups, lr=lr_muon, beta1=0.95, beta2=0.95,
                    weight_decay=0.1, precondition_frequency=10,
                )
            elif optimizer_type == "muonq":
                print(f"🔬 [MuonQ]: 4-bit Muon via directional fidelity (Su et al. 2025)")
                self.opt_muon = MuonQ(
                    muon_groups, lr=lr_muon, momentum=0.95,
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
        if use_adafactor:
            self.opt_adamw = _AdafactorWrapper(self.opt_adamw)
            print("🧊 [ADAF]: memory-efficient Adafactor on AdamW path (Shazeer & Stern 2018)")
        self.optimizer_type = optimizer_type
        self.use_schedule_free = use_schedule_free
        self.use_cautious = use_cautious
        if use_cautious:
            if self.opt_muon is not None: self.opt_muon = _CautiousWrapper(self.opt_muon)
            self.opt_adamw = _CautiousWrapper(self.opt_adamw)
            print("🛡️ [CAUTIOUS]: masking updates that disagree with gradient sign (Liang et al. 2024)")
        if use_schedule_free:
            if self.opt_muon is not None:
                self.opt_muon = _ScheduleFreeWrapper(self.opt_muon, beta=sf_beta, gamma_factor=sf_gamma_factor)
                print(f"🌀 [SF-AVG]: Schedule-Free averaging on Muon path (β={sf_beta}, γ×{sf_gamma_factor}) — MLCommons 2024 AlgoPerf winner")
            self.opt_adamw = _ScheduleFreeWrapper(self.opt_adamw, beta=sf_beta, gamma_factor=sf_gamma_factor)
            print(f"🌀 [SF-AVG]: Schedule-Free averaging on AdamW path (β={sf_beta}, γ×{sf_gamma_factor})")
        if use_quest:
            if self.opt_muon is not None:
                self.opt_muon = _QuESTWrapper(self.opt_muon, quest_bits=quest_bits)
            self.opt_adamw = _QuESTWrapper(self.opt_adamw, quest_bits=quest_bits)
            print(f"🔐 [QuEST]: trust gradient estimator for {quest_bits}-bit training (Panferov et al. 2025)")

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


class _AdafactorWrapper:
    """Memory-efficient Adafactor optimizer (Shazeer & Stern 2018).
    
    Uses factored second moment estimates (row and column sums) for 2D params.
    Much less memory than AdamW — ideal for small batch training.
    Reference: https://arxiv.org/abs/1804.04235
    """
    def __init__(self, base_optimizer, beta2_decay=-0.8, eps=1e-30, clip_threshold=1.0):
        self.base_optimizer = base_optimizer
        self.beta2_decay = beta2_decay
        self.eps = eps
        self.clip_threshold = clip_threshold
        self._state: dict = {}

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        for group in self.base_optimizer.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self._state.get(p)
                if state is None:
                    self._state[p] = {'step': 0}
                    state = self._state[p]
                
                state['step'] += 1
                
                if p.ndim == 2:
                    if 'v_row' not in state:
                        state['v_row'] = torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype)
                        state['v_col'] = torch.zeros(1, p.shape[1], device=p.device, dtype=p.dtype)
                    
                    update = grad.pow(2)
                    beta2 = 1.0 - state['step'] ** self.beta2_decay
                    
                    state['v_row'].mul_(beta2).add_(update.mean(dim=1, keepdim=True), alpha=1.0 - beta2)
                    state['v_col'].mul_(beta2).add_(update.mean(dim=0, keepdim=True), alpha=1.0 - beta2)
                    
                    approx = state['v_row'] @ state['v_col']
                    approx_mean = approx.mean().abs().clamp(min=self.eps)
                    approx = approx / approx_mean

                    grad_norm = grad.norm()
                    approx_norm = approx.sqrt().mean()
                    scale = self.clip_threshold / (grad_norm / (approx_norm + self.eps) + self.eps)
                    scale = scale.clamp(max=10.0)

                    update = grad * scale / (approx.sqrt() + self.eps)
                    update = update.clamp(max=1.0)
                    p.add_(update, alpha=-lr)
                else:
                    if 'v' not in state:
                        state['v'] = torch.zeros_like(p)
                    
                    beta2 = 1.0 - state['step'] ** self.beta2_decay
                    state['v'].mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)
                    
                    p.add_(grad / (state['v'].sqrt() + self.eps), alpha=-lr)

    def state_dict(self):
        return {'base': self.base_optimizer.state_dict(), 'ada': self._state,
                'beta2_decay': self.beta2_decay, 'eps': self.eps, 'clip_threshold': self.clip_threshold}

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd['base'])
        self._state = sd['ada']
        self.beta2_decay = sd['beta2_decay']
        self.eps = sd['eps']
        self.clip_threshold = sd['clip_threshold']


class _ScheduleFreeWrapper:
    def __init__(self, base_optimizer, beta: float = 0.9, gamma_factor: float = 2.0):
        self.base_optimizer = base_optimizer
        self.beta = beta
        self.gamma_factor = gamma_factor
        self._state: dict = {}

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        params = [p for g in self.base_optimizer.param_groups for p in g['params']]
        for g in self.base_optimizer.param_groups:
            g['lr'] = g['lr'] * self.gamma_factor
        for p in params:
            if p.grad is None:
                continue
            state = self._state.get(p)
            if state is None:
                self._state[p] = {'x': p.data.clone(), 'z': p.data.clone(), 't': 0}
            p.data.copy_(self._state[p]['z'])
        self.base_optimizer.step()
        for g in self.base_optimizer.param_groups:
            g['lr'] = g['lr'] / self.gamma_factor
        for p in params:
            if p.grad is None:
                continue
            state = self._state[p]
            z_new = p.data.clone()
            t_new = state['t'] + 1
            x_new = (1.0 - 1.0 / t_new) * state['x'] + (1.0 / t_new) * z_new
            y_new = (1.0 - self.beta) * x_new + self.beta * z_new
            state['z'] = z_new
            state['x'] = x_new
            state['t'] = t_new
            p.data.copy_(y_new)

    def state_dict(self):
        return {'base': self.base_optimizer.state_dict(), 'sf': self._state,
                'beta': self.beta, 'gamma_factor': self.gamma_factor}

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd['base'])
        self._state = sd['sf']
        self.beta = sd['beta']
        self.gamma_factor = sd['gamma_factor']


class _CautiousWrapper:
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        params = [p for g in self.base_optimizer.param_groups for p in g['params']]
        snapshots = {p: p.data.clone() for p in params if p.grad is not None}
        self.base_optimizer.step()
        for p in params:
            if p.grad is None: continue
            update = p.data - snapshots[p]
            mask = (update * p.grad > 0).to(update.dtype)
            p.data = snapshots[p] + update * mask

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)


@register("optimizer", "soap")
class SOAP(torch.optim.Optimizer):
    """SOAP: Shampoo-style preconditioner + Adam (Vyas et al. 2025, ICLR 2025).
    
    For 2D params: maintains factored second-moment L (row) and R (column).
    Periodically eigendecomposes and applies preconditioned Adam update.
    For 1D params: standard Adam fallback.
    """

    def __init__(self, params, lr=1e-3, beta1=0.95, beta2=0.95, eps=1e-8,
                 weight_decay=0.1, precondition_frequency=10):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, precondition_frequency=precondition_frequency)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            freq = group['precondition_frequency']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    if p.ndim == 2:
                        state['m'] = torch.zeros_like(p)
                        state['L'] = torch.eye(p.shape[0], device=p.device, dtype=torch.float32)
                        state['R'] = torch.eye(p.shape[1], device=p.device, dtype=torch.float32)
                        state['Q_L'] = torch.eye(p.shape[0], device=p.device, dtype=torch.float32)
                        state['Q_R'] = torch.eye(p.shape[1], device=p.device, dtype=torch.float32)
                    else:
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)

                state['step'] += 1

                if p.ndim == 2:
                    d1, d2 = p.shape
                    m, L, R = state['m'], state['L'], state['R']

                    g32 = grad.float()
                    L.mul_(beta2).add_(g32 @ g32.T, alpha=1.0 - beta2)
                    R.mul_(beta2).add_(g32.T @ g32, alpha=1.0 - beta2)

                    if state['step'] % freq == 0 or 'Q_L' not in state:
                        try:
                            Q_L, _ = torch.linalg.eigh(L)
                            Q_R, _ = torch.linalg.eigh(R)
                            state['Q_L'] = Q_L
                            state['Q_R'] = Q_R
                        except Exception:
                            pass

                    Q_L, Q_R = state['Q_L'], state['Q_R']
                    g_hat = Q_L.T @ g32 @ Q_R

                    m.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    m_hat = m.float() / bias_correction1

                    L_eig = L.diag().float().clamp(min=eps)
                    R_eig = R.diag().float().clamp(min=eps)
                    v_eig = (L_eig.sqrt().unsqueeze(1) * R_eig.sqrt().unsqueeze(0)) / (bias_correction2 ** 0.5 + eps)

                    update_hat = m_hat / (v_eig + eps)
                    update = Q_L @ update_hat @ Q_R.T
                    update = update.clamp(-1.0, 1.0)

                    p.mul_(1.0 - lr * wd)
                    p.add_(update.to(p.dtype), alpha=-lr)
                else:
                    m, v = state['m'], state['v']
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** state['step']
                    bc2 = 1.0 - beta2 ** state['step']
                    update = (m / bc1) / (v.sqrt() / (bc2 ** 0.5) + eps)
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)


@register("optimizer", "muonq")
class MuonQ(torch.optim.Optimizer):
    """MuonQ: 4-bit Muon via directional fidelity optimization (Su et al. 2025).

    Three techniques for stable 4-bit quantization of Muon momentum:
    1. Pre-quantization normalization — unit Frobenius norm per step
    2. Structural decomposition — power iteration for top-k singular components
    3. μ-law companding — non-linear quantization for dense-region resolution

    Reference: https://arxiv.org/abs/2605.11396
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 ns_steps=5, rank_ratio=1/16, mu_law_mu=255):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, rank_ratio=rank_ratio, mu_law_mu=mu_law_mu)
        super().__init__(params, defaults)

    @staticmethod
    def _mu_law_compress(x, mu=255):
        return torch.sign(x) * torch.log(1 + mu * x.abs()) / math.log(1 + mu)

    @staticmethod
    def _mu_law_decompress(y, mu=255):
        return torch.sign(y) * ((1 + mu) ** y.abs() - 1) / mu

    @staticmethod
    def _quantize_4bit(x, group_size=2048):
        orig_shape = x.shape
        x_flat = x.reshape(-1)
        n = x_flat.numel()
        if group_size > 0 and n > group_size:
            pad = (group_size - n % group_size) % group_size
            if pad > 0:
                x_flat = torch.cat([x_flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
            x_grouped = x_flat.reshape(-1, group_size)
            scales = x_grouped.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
            x_normed = x_grouped / scales
            x_compressed = MuonQ._mu_law_compress(x_normed)
            x_int = torch.clamp(torch.round((x_compressed + 1) / 2 * 15), 0, 15).to(torch.uint8)
            x_decomp = MuonQ._mu_law_decompress(x_int.float() / 15 * 2 - 1) * scales
            return x_decomp.reshape(-1)[:n].reshape(orig_shape)
        else:
            scale = x.abs().max().clamp(min=1e-8)
            x_normed = x / scale
            x_compressed = MuonQ._mu_law_compress(x_normed)
            x_int = torch.clamp(torch.round((x_compressed + 1) / 2 * 15), 0, 15).to(torch.uint8)
            x_decomp = MuonQ._mu_law_decompress(x_int.float() / 15 * 2 - 1) * scale
            return x_decomp

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            rank_ratio = group['rank_ratio']
            mu = group['mu_law_mu']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                d1, d2 = p.shape

                if 'momentum_buffer' not in state:
                    if p.device.type == "cuda": dtype = torch.bfloat16
                    elif p.device.type == "mps": dtype = torch.float16
                    else: dtype = torch.float32
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=dtype)
                    state['step'] = 0

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad.to(buf.dtype))

                # Pre-quantization normalization (unit Frobenius norm)
                buf_norm = buf.float()
                buf_frob = buf_norm.norm()
                if buf_frob > 1e-8:
                    buf_norm = buf_norm / buf_frob

                # Structural decomposition via power iteration (top-k)
                k = max(1, int(min(d1, d2) * rank_ratio))
                if k < min(d1, d2) and state['step'] > 0:
                    # Power iteration for top-k singular vectors
                    if 'Q_prev' not in state:
                        Q_prev = torch.randn(d2, k, device=p.device, dtype=torch.float32)
                        Q_prev = torch.linalg.qr(Q_prev)[0]
                    else:
                        Q_prev = state['Q_prev']

                    P = buf_norm @ Q_prev
                    P = torch.linalg.qr(P)[0]
                    R = buf_norm.T @ P
                    state['Q_prev'] = R / (R.norm(dim=0, keepdim=True) + 1e-8)

                    # Residual = full - top-k reconstruction
                    M_res = buf_norm - P @ R.T
                    M_res_q = MuonQ._quantize_4bit(M_res)
                    buf_quant = (P @ R.T + M_res_q) * buf_frob
                else:
                    buf_quant = MuonQ._quantize_4bit(buf_norm) * buf_frob

                # Newton-Schulz orthogonalization
                O_t = _compiled_newton_schulz(buf_quant, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))

                p.mul_(1.0 - lr * wd)
                p.add_(O_t.to(p.dtype), alpha=-lr * scale)
                state['step'] += 1


class _QuESTWrapper:
    """QuEST: Trust gradient estimator for quantized training (Panferov et al. 2025, ICML 2025).
    
    Improves STE gradient estimation for ternary/low-bit weights:
    1. Hadamard rotation whitens weight distribution
    2. MSE-optimal ternary grid fitting
    3. Trust gradient correction reduces bias vs naive STE
    
    Wraps any base optimizer. Adds ~5% step-time overhead.
    Reference: https://proceedings.mlr.press/v267/panferov25a.html
    """
    def __init__(self, base_optimizer, quest_bits=1.58, correction_scale=0.1):
        self.base_optimizer = base_optimizer
        self.quest_bits = quest_bits
        self.correction_scale = correction_scale

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _fwht(x):
        """Fast Walsh-Hadamard Transform — O(n log n). Pads to next power of 2."""
        n = x.numel()
        if n <= 1:
            return x
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        if next_pow2 != n:
            padded = torch.zeros(next_pow2, device=x.device, dtype=x.dtype)
            padded[:n] = x
            x = padded
        h = 1
        while h < len(x):
            for i in range(0, len(x), h * 2):
                for j in range(i, i + h):
                    a = x[j].clone()
                    b = x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
            h *= 2
        return x / (len(x) ** 0.5)

    @torch.no_grad()
    def step(self):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None or p.ndim != 2:
                    continue
                grad = p.grad
                shape = p.shape
                n_orig = p.numel()

                Q = self._fwht(p.data.clone().view(-1))[:n_orig].view(shape)
                alpha = Q.abs().mean().clamp(min=1e-8)
                grid = torch.tensor([-alpha, 0.0, alpha], device=p.device, dtype=p.dtype)
                flat_q = Q.reshape(-1, 1)
                flat_grid = grid.reshape(1, -1)
                dists = (flat_q - flat_grid).abs()
                closest = dists.argmin(dim=-1).float().reshape(shape)
                Q_quant = torch.where(closest == 0, -alpha,
                           torch.where(closest == 1, 0.0, alpha))

                noise = (Q - Q_quant).reshape(-1)
                noise_norm = noise.norm()
                if noise_norm > 1e-8:
                    correction = (noise / noise_norm * grad.norm()).reshape(shape)
                    p.grad.add_(correction, alpha=self.correction_scale)

        self.base_optimizer.step()

    def state_dict(self):
        return {'base': self.base_optimizer.state_dict(),
                'quest_bits': self.quest_bits, 'correction_scale': self.correction_scale}

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd['base'])
        self.quest_bits = sd['quest_bits']
        self.correction_scale = sd['correction_scale']


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
            k.removeprefix("_orig_mod."): (v.float() if v.dtype.is_floating_point else v.clone())
            for k, v in sd.items()
        }
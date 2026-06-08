"""
⚙️ busel LAYERS v4.7 - Autocast Safe, SwishGLU, Fused RMSNorm & H_BitLinear (BitNet v2)
"""
import torch
import torch.nn as nn
import math

_BITLINEAR_CONFIG = {"use_tequila": False, "tequila_lambda": 1e-3, "hestia_temperature": None}

def configure_bitlinear(use_tequila: bool = False, tequila_lambda: float = 1e-3,
                        hestia_temperature=None):
    _BITLINEAR_CONFIG["use_tequila"] = use_tequila
    _BITLINEAR_CONFIG["tequila_lambda"] = tequila_lambda
    _BITLINEAR_CONFIG["hestia_temperature"] = hestia_temperature

def nvtx_range_push(name: str):
    if torch.cuda.is_available(): torch.cuda.nvtx.range_push(name)
def nvtx_range_pop():
    if torch.cuda.is_available(): torch.cuda.nvtx.range_pop()

def fast_walsh_hadamard_transform(x):
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.view(-1, D)
    N_flat = x_flat.shape[0]
    power_of_2 = 2 ** math.ceil(math.log2(D))
    if D != power_of_2:
        x_flat = torch.nn.functional.pad(x_flat, (0, power_of_2 - D))
    h = 1
    while h < power_of_2:
        x_flat = x_flat.view(N_flat, -1, h * 2)
        x1 = x_flat[..., :h]
        x2 = x_flat[..., h:]
        x_flat = torch.cat([x1 + x2, x1 - x2], dim=-1)
        h *= 2
    x_flat = x_flat.view(N_flat, power_of_2) / math.sqrt(power_of_2)
    if D != power_of_2:
        x_flat = x_flat[..., :D]
    return x_flat.view(orig_shape)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output


class HestiaQuantize(torch.autograd.Function):
    """Hestia: temperature-controlled softmax relaxation for ternary quantization.

    Instead of hard round (STE), uses softmax expectation over {-1, 0, 1} codebook.
    Temperature anneals from high (soft) to low (hard) during training.
    Provides exact gradient fidelity — no deadzone, no gradient mismatch.

    Reference: Wang et al. 2026 (arXiv:2601.20745)
    """
    @staticmethod
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x, temperature)
        if temperature.item() < 0.01:
            return torch.clamp(torch.round(x), -1, 1)
        codebook = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
        logits = -((x.unsqueeze(-1) - codebook) ** 2) / (temperature + 1e-8)
        probs = torch.softmax(logits, dim=-1)
        return (probs * codebook).sum(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        x, temperature = ctx.saved_tensors
        if temperature.item() < 0.01:
            return grad_output, None
        codebook = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
        logits = -((x.unsqueeze(-1) - codebook) ** 2) / (temperature + 1e-8)
        probs = torch.softmax(logits, dim=-1)
        dx = (probs * (codebook.unsqueeze(0) - x.unsqueeze(-1)) * 2 / (temperature + 1e-8)).sum(dim=-1)
        grad_x = grad_output * dx
        d_temp = (probs * (codebook.unsqueeze(0) - x.unsqueeze(-1)) ** 2 / (temperature + 1e-8) ** 2).sum(dim=-1)
        grad_temp = (grad_output * d_temp).sum()
        return grad_x, grad_temp

class BitLinear_a4_8(nn.Linear):
    """Ternary linear layer with INT4/INT8 activation quantization.

    v7.0: Tequila deadzone reactivation (Huang et al. 2025, ICLR 2026).
    v7.0: Hestia temperature-controlled softmax relaxation (Wang et al. 2026).
    """
    def __init__(self, in_features, out_features, is_intermediate=False,
                 topk_ratio=0.5, use_tequila=False, tequila_lambda=1e-3,
                 hestia_temperature=None):
        super().__init__(in_features, out_features, bias=False)
        self.is_intermediate = is_intermediate
        self.topk_ratio = topk_ratio
        self.use_tequila = use_tequila
        self.tequila_lambda = tequila_lambda
        self.hestia_temperature = hestia_temperature
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean().detach() + 1e-5
        w_scaled = w / alpha
        w_clipped = torch.clamp(w_scaled, -1, 1)

        use_tequila = self.use_tequila or _BITLINEAR_CONFIG["use_tequila"]
        tequila_lambda = self.tequila_lambda or _BITLINEAR_CONFIG["tequila_lambda"]
        temp = self.hestia_temperature if self.hestia_temperature is not None else _BITLINEAR_CONFIG["hestia_temperature"]

        if temp is not None and temp.item() > 0 if hasattr(temp, 'item') else temp is not None and temp > 0:
            temp_val = temp if isinstance(temp, torch.Tensor) else torch.tensor(temp, device=w.device, dtype=w.dtype)
            w_quant = HestiaQuantize.apply(w_clipped, temp_val)
        else:
            w_quant = w_clipped + (RoundSTE.apply(w_clipped) - w_clipped)

        tequila_bias = None
        if use_tequila and not self.is_intermediate:
            deadzone_mask = (w_scaled.abs() < 0.5).to(w.dtype)
            tequila_bias = tequila_lambda * (w * deadzone_mask).sum(dim=-1)

        if not self.is_intermediate:
            beta = x.abs().mean(dim=-1, keepdim=True).detach() + 1e-5
            x_scaled = x * (2.6457 / beta)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -8, 7)) - x_scaled)
            out = nn.functional.linear(x_quant, w_quant)
            out = out * (alpha * beta / 2.6457)
            if tequila_bias is not None:
                out = out + tequila_bias
            return out
        else:
            gamma = x.abs().max(dim=-1, keepdim=True)[0].detach() + 1e-5
            x_scaled = x * (127.0 / gamma)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -128, 127)) - x_scaled)
            if self.topk_ratio < 1.0:
                k = int(x.shape[-1] * self.topk_ratio)
                mask = torch.zeros_like(x_quant)
                topk_vals, _ = torch.topk(x_quant.abs(), k, dim=-1)
                mask[x_quant.abs() >= topk_vals[..., -1:]] = 1.0
                x_quant = x_quant * mask
            out = nn.functional.linear(x_quant, w_quant)
            return out * (alpha * gamma / 127.0)

class H_BitLinear(BitLinear_a4_8):
    def forward(self, x):
        x_rotated = fast_walsh_hadamard_transform(x)
        return super().forward(x_rotated)

class LearnableClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bounds):
        ctx.save_for_backward(x, bounds)
        return torch.clamp(x, -bounds, bounds)
    @staticmethod
    def backward(ctx, grad_output):
        x, bounds = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_bounds = grad_output.clone()
        grad_bounds = (grad_bounds * (x > bounds).float()) - (grad_bounds * (x < -bounds).float())
        sum_dims = list(range(grad_bounds.ndim - 1))
        if sum_dims: grad_bounds = grad_bounds.sum(dim=sum_dims)
        return grad_x, grad_bounds

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if hasattr(torch.nn.functional, "rms_norm"):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight.to(x.dtype)

class SwishGLUClamped(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w_gate_up = BitLinear_a4_8(d_model, 2 * d_ffn)
        self.w_down = H_BitLinear(d_ffn, d_model, is_intermediate=True)
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)
        self.down_norm = RMSNorm(d_ffn) # 🎯 ИСПРАВЛЕНИЕ: RMSNorm перед H-BitLinear

    def forward(self, x):
        gate_up = self.w_gate_up(x)
        gate_raw, up = gate_up.chunk(2, dim=-1)
        gate_swish = gate_raw * torch.sigmoid(gate_raw)
        gate = LearnableClampSTE.apply(gate_swish, self.clipping_bounds)
        return self.w_down(self.down_norm(gate * up)) # 🎯 ИСПРАВЛЕНИЕ
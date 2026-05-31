"""
⚙️ BYSEL LAYERS v4.7 - Autocast Safe, SwishGLU, Fused RMSNorm & H-BitLinear (BitNet v2)
"""
import torch
import torch.nn as nn
import math

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

class BitLinear_a4_8(nn.Linear):
    def __init__(self, in_features, out_features, is_intermediate=False, topk_ratio=0.5):
        super().__init__(in_features, out_features, bias=False)
        self.is_intermediate = is_intermediate
        self.topk_ratio = topk_ratio
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean().detach() + 1e-5
        w_scaled = w / alpha
        w_clipped = torch.clamp(w_scaled, -1, 1)
        w_quant = w_clipped + (RoundSTE.apply(w_clipped) - w_clipped)

        if not self.is_intermediate:
            beta = x.abs().mean(dim=-1, keepdim=True).detach() + 1e-5
            x_scaled = x * (2.6457 / beta)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -8, 7)) - x_scaled)
            out = nn.functional.linear(x_quant, w_quant)
            return out * (alpha * beta / 2.6457)
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
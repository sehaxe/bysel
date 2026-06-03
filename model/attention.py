"""
💡 busel ATTENTION v5.2 - Gated DeltaNet-2 & MLA (Stabilized Broadcasting)
Интегрирован раздельный закон стирания и записи GDN-2,
когерентный логарифмический распад alpha (Eq. 12) с выверенным бродкастом.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, H_BitLinear, RMSNorm, nvtx_range_push, nvtx_range_pop
from busel_registry import register

# 🎯 Нативный импорт официального ядра GDN-2 из flash-linear-attention
try:
    from fla.ops.gdn2 import chunk_gdn2
    HAS_FLA_GDN2 = True
except ImportError:
    HAS_FLA_GDN2 = False


@torch.jit.script
def stable_gdn2_recurrent_jit(q, k, v, b, w, alpha):
    """
    Высокооптимизированный JIT-компилированный рекуррентный цикл GDN-2 (NVIDIA, 2026).
    Реализует раздельные канальные гейты стирания (b) и записи (w) с поканальным затуханием (alpha).
    """
    B, T, H, dk = q.size()
    dv = v.size(-1)
    
    # S: [B, H, dk, dv]
    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=q.dtype)
    out = torch.zeros(B, T, H, dv, device=q.device, dtype=q.dtype)
    
    for t in range(T):
        q_t = q[:, t]          # [B, H, dk]
        k_t = k[:, t]          # [B, H, dk]
        v_t = v[:, t]          # [B, H, dv]
        b_t = b[:, t]          # [B, H, dk]
        w_t = w[:, t]          # [B, H, dv]
        alpha_t = alpha[:, t]  # [B, H, dk]
        
        # 1. Затухание прошлого состояния по канальной маске альфа: S_bar = D_t * S_{t-1}
        S_bar = S * alpha_t.unsqueeze(-1)
        
        # 2. Вычисление проекции стирания: r_t = S_bar^T * (b_t ⊙ k_t)
        erase_key = b_t * k_t
        r_t = torch.einsum('bhkd,bhk->bhd', S_bar, erase_key)
        
        # 3. Вычисление целевой записи: z_t = w_t ⊙ v_t
        z_t = w_t * v_t
        
        # 4. Обновление состояния: S_t = S_bar + k_t * (z_t - r_t)^T
        update = z_t - r_t
        outer = k_t.unsqueeze(-1) * update.unsqueeze(-2)
        S = S_bar + outer
        
        # 5. Считывание выхода: o_t = S_t^T * q_t
        out_t = torch.einsum('bhkd,bhk->bhd', S, q_t)
        out[:, t] = out_t
        
    return out.reshape(B, T, -1)


@register("attention", "gdn2")
class BulbaGDN2SeRoPEBlock(nn.Module):
    def __init__(self, d_model=1536, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.q_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.k_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.v_proj = BitLinear_a4_8(d_model, n_heads * self.d_v)
        
        # Каузальные depthwise свертки (NVIDIA GDN-2 Spec)
        self.q_conv = nn.Conv1d(n_heads * self.d_k, n_heads * self.d_k, kernel_size=4, groups=n_heads * self.d_k, padding=0)
        self.k_conv = nn.Conv1d(n_heads * self.d_k, n_heads * self.d_k, kernel_size=4, groups=n_heads * self.d_k, padding=0)
        self.v_conv = nn.Conv1d(n_heads * self.d_v, n_heads * self.d_v, kernel_size=4, groups=n_heads * self.d_v, padding=0)
        
        self.b_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.w_proj = nn.Linear(d_model, n_heads * self.d_v)
        
        # ЛОГАРИФМИЧЕСКИЙ РАСПАД NVIDIA GDN-2 (Формула 12)
        self.alpha_proj = nn.Linear(d_model, n_heads * self.d_k)
        # Обучаемый вектор масштаба логарифмического затухания, инициализируемый отрицательным числом
        self.alpha_a = nn.Parameter(torch.ones(n_heads, 1) * -3.0)
        
        # Активируем Triton-ядро только если оно есть в библиотеке и мы на GPU с поддержкой CUDA
        if HAS_FLA_GDN2 and torch.cuda.is_available():
            self.use_fla = True
        else:
            self.use_fla = False
            
        # Низкоранговый выходной гейт (Output Gating — Формула 10)
        self.g_proj_down = BitLinear_a4_8(d_model, d_model // 4)
        self.g_proj_up = BitLinear_a4_8(d_model // 4, d_model)
        self.out_norm = RMSNorm(d_model)
        
        # o_proj заменена на H_BitLinear по спецификации BitNet v2
        self.o_proj = H_BitLinear(d_model, d_model)
        self.register_buffer("freqs", 10000 ** (-torch.arange(0, self.d_k, 2).float() / self.d_k))

    def apply_serope(self, T, q, k):
        B, _, H, _ = q.shape
        q_real, q_imag = q[..., 0::2], q[..., 1::2]
        k_real, k_imag = k[..., 0::2], k[..., 1::2]
        
        angles = torch.arange(T, device=q.device).view(1, T, 1, 1) * self.freqs.view(1, 1, 1, -1)
        cos, sin = torch.cos(angles), torch.sin(angles)
        
        q_out = torch.zeros_like(q)
        k_out = torch.zeros_like(k)
        q_out[..., 0::2], q_out[..., 1::2] = q_real * cos - q_imag * sin, q_real * sin + q_imag * cos
        k_out[..., 0::2], k_out[..., 1::2] = k_real * cos + k_imag * sin, -k_real * sin + k_imag * cos
        return q_out, k_out

    def forward(self, x):
        nvtx_range_push("busel_GDN2_SeRoPE_Forward")
        B, T, C = x.shape
        
        # 1. Линейные проекции и каузальные свертки
        q_proj = self.q_proj(x).transpose(1, 2)
        q_conv = self.q_conv(F.pad(q_proj, (3, 0)))  # Левосторонний причинный паддинг
        q = F.silu(q_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_k)
        
        k_proj = self.k_proj(x).transpose(1, 2)
        k_conv = self.k_conv(F.pad(k_proj, (3, 0)))
        k = F.silu(k_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_k)
        
        v_proj = self.v_proj(x).transpose(1, 2)
        v_conv = self.v_conv(F.pad(v_proj, (3, 0)))
        v = F.silu(v_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_v)
        
        # L2-нормализация ключей и запросов для рекуррентной стабильности
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        q, k = self.apply_serope(T, q, k)
        
        b = torch.sigmoid(self.b_proj(x)).view(B, T, self.n_heads, self.d_k)
        w = torch.sigmoid(self.w_proj(x)).view(B, T, self.n_heads, self.d_v)
        
        # 🎯 ВЫЧИСЛЕНИЕ ЛОГАРИФМИЧЕСКОГО РАСПАДА (NVIDIA GDN-2 Spec — Формула 12):
        # view(1, 1, self.n_heads, 1) выравнивает размерность альфа-а под бродкаст на [B, T, n_heads, d_k]
        alpha_proj = self.alpha_proj(x).view(B, T, self.n_heads, self.d_k)
        g_t = -torch.exp(self.alpha_a).view(1, 1, self.n_heads, 1) * F.softplus(alpha_proj)
        alpha = torch.exp(g_t)
        
        if self.use_fla:
            # 🚀 СВЕРХБЫСТРЫЙ И КОРРЕКТНЫЙ TRITON-ИНФЕРЕНС GDN-2 (PR #920):
            # Передаем q, k, v, логарифмический распад g_t, гейт очистки b и гейт записи w.
            # По умолчанию в FLA используется формат [B, T, H, D]
            out, _ = chunk_gdn2(
                q, 
                k, 
                v, 
                g_t,        # Передаем логарифмический распад g_t
                b,          # Передаем гейт стирания b
                w,          # Передаем гейт записи w
                scale=1.0 / (self.d_k ** 0.5)
            )
            out = out.reshape(B, T, -1)
        else:
            out = stable_gdn2_recurrent_jit(q, k, v, b, w, alpha)
            
        # Применение выходного гейта и RMSNorm (Формула 10)
        gate = torch.sigmoid(self.g_proj_up(self.g_proj_down(x)))
        out_gated = self.out_norm(out) * gate
        
        res = self.o_proj(out_gated)
        nvtx_range_pop()
        return res


@register("attention", "mla")
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=1536, n_heads=12, d_c=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_c = d_c
        self.d_v = d_model // n_heads
        
        self.kv_compress = BitLinear_a4_8(d_model, d_c)
        self.kv_norm = RMSNorm(d_c)
        self.k_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        self.v_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        
        self.q_compress = BitLinear_a4_8(d_model, d_c)
        self.q_norm = RMSNorm(d_c)
        self.q_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        
        self.out_norm = RMSNorm(n_heads * self.d_v)
        # o_proj заменена на H_BitLinear по спецификации BitNet v2
        self.o_proj = H_BitLinear(n_heads * self.d_v, d_model)

    def forward(self, x):
        nvtx_range_push("busel_MLA_Forward")
        B, T, C = x.shape
        kv_latent = self.kv_norm(self.kv_compress(x))
        k = self.k_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        v = self.v_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        
        q_latent = self.q_norm(self.q_compress(x))
        q = self.q_decompress(q_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        
        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(self.out_norm(context))
        nvtx_range_pop()
        return out
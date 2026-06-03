"""
FastBLT: Byte-Level Tokenizer (безтокенный ввод с причинной сверткой и гейтированием)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import RMSNorm, nvtx_range_push, nvtx_range_pop

class StridedFastBLTPatcher(nn.Module):
    def __init__(self, d_model=768, d_byte=128, stride=4, kernel_size=5):
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        self.d_byte = d_byte
        self.kernel_size = kernel_size
        
        self.embed_weight = nn.Parameter(torch.randn(259, d_byte) * 0.02)
        
        # 🎯 ИСПРАВЛЕНИЕ: Нелинейный гейт (Mini-SwishGLU) для фильтрации синтаксического шума
        self.gate_proj_down = nn.Linear(d_byte, max(1, d_byte // 4))
        self.gate_proj_up = nn.Linear(max(1, d_byte // 4), d_byte)
        
        self.conv = nn.Conv1d(d_byte, d_model, kernel_size=kernel_size, stride=stride, padding=0)
        self.norm = RMSNorm(d_model)

    def forward(self, byte_ids):
        nvtx_range_push("busel_Byte_Patching_Forward")
        byte_ids_device = byte_ids.to(self.embed_weight.device)
        x = F.embedding(byte_ids_device, self.embed_weight)
        
        # 🎯 ИСПРАВЛЕНИЕ: Применяем нелинейный Swish-гейт
        gate = torch.sigmoid(self.gate_proj_up(F.silu(self.gate_proj_down(x))))
        x = x * gate
        
        x = x.transpose(1, 2)
        x_padded = F.pad(x, (self.kernel_size - 1, 0))
        patches = self.conv(x_padded)
        patches = patches.transpose(1, 2)
        out = self.norm(patches)
        nvtx_range_pop()
        return out
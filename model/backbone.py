"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL BACKBONE v5.2 - Mathematically Exact mAR (DeepSeek mHC + Kimi)      ║
║                                                                           ║
║ Компоненты:                                                               ║
║  • ByselDecoderLayer: Returns pure, unaccumulated f_l(h_l) transformation  ║
║  • ManifoldConstrainedAttnRes (mAR): Exact Birkhoff Attention over depth  ║
║  • ByselModel: proper sequential routing of unaccumulated layer outputs   ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8, RMSNorm, nvtx_range_push, nvtx_range_pop
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE


class ManifoldConstrainedAttnRes(nn.Module):
    """
    mAR: Скрещение Kimi Attention Residuals и DeepSeek mHC.
    Вычисляет чистую взвешенную сумму по всем предшествующим неаккумулированным выходам v_i.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = RMSNorm(d_model)
        nn.init.zeros_(self.proj.weight)

    def forward(self, current_x, all_prev_outputs):
        # Если это первый слой, доступен только один источник (эмбеддинги h_1)
        if len(all_prev_outputs) <= 1:
            return all_prev_outputs[0]
        
        # Вычисляем сырые логиты внимания (AttnRes) по всем неаккумулированным выходам v_i
        logits_list = []
        proj_weight = self.proj.weight.squeeze()
        
        for prev_x in all_prev_outputs:
            K_part = self.norm(prev_x)
            logit_part = torch.einsum('d, b t d -> b t', proj_weight, K_part)
            logits_list.append(logit_part)
        
        M = torch.stack(logits_list, dim=0) # [L, B, T]
        
        # Проекция матрицы внимания на Birkhoff Polytope (DeepSeek mHC)
        M_stable = M - M.max(dim=0, keepdim=True)[0]
        M = torch.exp(M_stable)
        for _ in range(3):
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Чистая взвешенная сумма h_l = \sum \alpha_i * v_i (Формула 4)
        h = torch.zeros_like(current_x)
        for l in range(len(all_prev_outputs)):
            h = h + M[l].unsqueeze(-1) * all_prev_outputs[l]
            
        return h


class ByselDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden, num_experts, is_global=False, capacity_factor=1.0):
        super().__init__()
        self.mod_router = MoDSequenceRouter(d_model, capacity_factor=capacity_factor)
        
        if is_global:
            self.attn = MultiHeadLatentAttention(d_model, n_heads)
        else:
            self.attn = BulbaGDN2SeRoPEBlock(d_model, n_heads)
        
        self.moe = BulbaTernaryTitanMoE(d_model, expert_hidden, num_experts=num_experts)
        
        self.attn_norm = RMSNorm(d_model)
        self.moe_norm = RMSNorm(d_model)

    def forward(self, x, progress=0.0):
        # MoD-маршрутизация при емкости >= 1.0
        if self.mod_router.capacity_factor >= 1.0:
            attn_out = self.attn(self.attn_norm(x))
            moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
            
            # Слой возвращает строго чистый неаккумулированный выход преобразования f_l(h_l).
            return moe_out, aux_loss
            
        # Пакетная маршрутизация MoD при capacity_factor < 1.0
        B, T, C = x.shape
        mask, logits = self.mod_router(x)
        
        k = int(T * self.mod_router.capacity_factor)
        if k == 0:
            return torch.zeros_like(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
        active_tokens = x[mask].view(B, k, C)
        
        attn_out = self.attn(self.attn_norm(active_tokens))
        moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
        
        gated_out = moe_out * torch.sigmoid(logits[mask]).view(B, k, 1)
        
        out = torch.zeros_like(x)
        out[mask] = gated_out.view(-1, C)
        
        return out, aux_loss


class ByselMTP4Pipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_weight = nn.Parameter(
            torch.randn(config.vocab_size, config.d_model) * 0.02
        )
        
        self.projections = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.d_model) 
            for _ in range(3)
        ])
        
        self.heads = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.vocab_size) 
            for _ in range(4)
        ])

    def _embed_lookup(self, token_ids):
        return self.embed_weight[token_ids.to(self.embed_weight.device)]

    def forward(self, main_hidden_states, next_token_ids=None):
        logits_t1 = self.heads[0](main_hidden_states)
        
        if next_token_ids is None or any(t is None for t in next_token_ids):
            return logits_t1, None, None, None
        
        h_detached = main_hidden_states.detach()
        
        # t+2
        combined_t2 = self.projections[0](h_detached) + self._embed_lookup(next_token_ids[0])
        logits_t2 = self.heads[1](combined_t2)
        
        # t+3
        combined_t3 = self.projections[1](combined_t2) + self._embed_lookup(next_token_ids[1])
        logits_t3 = self.heads[2](combined_t3)
        
        # t+4
        combined_t4 = self.projections[2](combined_t3) + self._embed_lookup(next_token_ids[2])
        logits_t4 = self.heads[3](combined_t4)
        
        return logits_t1, logits_t2, logits_t3, logits_t4


class ByselModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        capacity = 1.0 
        
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            is_global = (l + 1) % 4 == 0
            self.layers.append(ByselDecoderLayer(
                config.d_model, 
                config.n_heads, 
                config.expert_hidden, 
                config.num_experts, 
                is_global=is_global,
                capacity_factor=capacity
            ))
        
        self.m_residuals = nn.ModuleList([
            ManifoldConstrainedAttnRes(config.d_model) 
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.d_model)
        self.mtp_pipeline = ByselMTP4Pipeline(config)
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False

    def forward(self, x, next_token_ids=None, progress=0.0):
        nvtx_range_push("ByselModel_Forward")
        
        # Инициализация источников AttnRes исходными эмбеддингами h_1
        prev_outputs = [x]
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            m_res = self.m_residuals[i]
            
            # Расчет входа h_i для текущего слоя через взвешивание mAR по всем прошлым выходам v_i
            h_i = m_res(x, prev_outputs)
            
            # Чекпоинтинг с поддержкой CUDA и MPS (через reentrant=False)
            if self.training and self.use_gradient_checkpointing and x.device.type in ["cuda", "mps"]:
                layer_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, h_i, progress,
                    use_reentrant=False
                )
            else:
                layer_out, aux_loss = layer(h_i, progress=progress)
            
            total_aux_loss += aux_loss
            # Накапливаем чистый неаккумулированный выход f_i(h_i)
            prev_outputs.append(layer_out)
            # Обновляем базовый указатель x для следующей итерации
            x = h_i
            
        # Финальное скрытое состояние h_{L+1} собирается как взвешенная сумма по ВСЕМ выходам
        final_hidden = self.final_norm(m_res(x, prev_outputs))
        mtp_outputs = self.mtp_pipeline(final_hidden, next_token_ids)
        
        nvtx_range_pop()
        return mtp_outputs, total_aux_loss
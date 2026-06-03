"""
⚙️ busel ROUTING v4.0 - Stable MoE & MoD
Интегрирован SwishGLUClamped в эксперты MoE и down-проекции переведены на H-BitLinear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, H_BitLinear, SwishGLUClamped, LearnableClampSTE, nvtx_range_push, nvtx_range_pop


class MoDSequenceRouter(nn.Module):
    def __init__(self, d_model, capacity_factor=0.25):
        super().__init__()
        self.router = nn.Linear(d_model, 1)
        self.capacity_factor = capacity_factor
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, x):
        nvtx_range_push("busel_MoD_Routing_Forward")
        B, T, C = x.shape
        k = int(T * self.capacity_factor)
        logits = self.router(x).squeeze(-1)
        
        _, topk_indices = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        
        nvtx_range_pop()
        return mask, logits


class BulbaTernaryTitanExpertFFN(nn.Module):
    """
    FFN-блок одного MoE-эксперта со слиянием проекций Gate-Up.
    Нативно использует класс SwishGLUClamped из BitNet v2.
    """
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.ffn = SwishGLUClamped(d_model, d_ffn)

    def forward(self, x):
        return self.ffn(x)


class BulbaTernaryTitanMoE(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=64, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Shared Experts
        self.shared_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn) 
            for _ in range(2)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn) 
            for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        self.w_gate_blackboard = BitLinear_a4_8(d_model, d_model)
        self.w_read_blackboard = BitLinear_a4_8(d_model, d_model)

    def forward(self, x, progress=0.0, aux_loss_weight=0.05, z_loss_weight=0.001):
        nvtx_range_push("busel_MoE_Experts_Forward")
        B, T, D = x.shape
        
        # Динамическое расписание штрафа роутера (MoE Scheduling):
        # На старте (progress < 0.1) даем свободу, к середине повышаем до 0.08 для строгой балансировки
        if progress < 0.1:
            current_aux_weight = 0.01
        else:
            current_aux_weight = min(0.08, 0.01 + 0.175 * (progress - 0.1))
        
        # 1. SHARED EXPERTS
        h_bb = (self.shared_experts[0](x) + self.shared_experts[1](x)) / 2.0
        
        # 2. BLACKBOARD MEMORY
        gate_signal = torch.sigmoid(self.w_gate_blackboard(x))
        read_signal = self.w_read_blackboard(h_bb)
        x_enriched = x + gate_signal * read_signal
        
        # 3. ANTICIPATORY ROUTING
        router_logits = self.router(x_enriched.detach()).to(dtype=x_enriched.dtype)
        
        if self.training:
            noise = torch.randn_like(router_logits) * 0.5
            router_logits = router_logits + noise
        
        z_loss = z_loss_weight * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        # 4. TOP-K SELECTION
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 5. ROUTED EXPERTS
        routed_output = torch.zeros_like(x_enriched)
        
        expert_masks = []
        expert_weights_list = []
        for i in range(self.num_experts):
            mask = (topk_indices == i)
            weight_sum = torch.where(
                mask, topk_weights, torch.zeros_like(topk_weights)
            ).sum(dim=-1)
            expert_masks.append(mask.any(dim=-1))
            expert_weights_list.append(weight_sum)
        
        for i in range(self.num_experts):
            mask = expert_masks[i]
            if mask.any():
                tokens = x_enriched[mask]
                out = self.routed_experts[i](tokens)
                weights = expert_weights_list[i][mask].unsqueeze(-1)
                routed_output[mask] = out * weights
        
        # 6. LOAD BALANCING LOSS (с учетом динамического веса)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = expert_masks[i].sum().float()
        
        f_i = tokens_per_expert / (B * T * self.top_k)
        P_i = routing_weights.mean(dim=(0, 1))
        
        load_balance_loss = current_aux_weight * self.num_experts * torch.sum(f_i * P_i)
        
        nvtx_range_pop()
        return h_bb + routed_output, load_balance_loss + z_loss
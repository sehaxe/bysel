"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ busel BACKBONE v5.4 - mAR: mHC (DeepSeek) + AttnRes (Kimi) — exact      ║
║                                                                           ║
║ Manifold-Constrained Attention Residuals combines:                        ║
║   • mHC:  n_hyper parallel streams, mixing H ∈ Birkhoff polytope         ║
║           via Sinkhorn-Knopp (restores identity-mapping property)         ║
║   • AttnRes: input-dependent H computed via multi-query attention         ║
║           (q from current input, k from each stream)                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
import math
import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8, RMSNorm, nvtx_range_push, nvtx_range_pop
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE


class ManifoldConstrainedAttnRes(nn.Module):
    """Manifold-Constrained Attention Residuals (mAR).

    Combines Kimi Attention Residuals (input-dependent attention over layer
    outputs) with DeepSeek mHC (Sinkhorn-Knopp projection of the mixing matrix
    onto the Birkhoff polytope of doubly-stochastic matrices).

    Maintains n_hyper parallel residual streams. At each call:
      1. Compute n queries from current input, n keys from the n streams
         (multi-query attention — AttnRes spirit).
      2. Build raw H_logits ∈ R^{n×n} per (B, T) via q·k + fixed identity
         bias (+5.0 on diagonal, mHC's identity-mapping property at init).
      3. Project to Birkhoff polytope via Sinkhorn-Knopp (mHC constraint).
      4. Mix the n streams with H, return the mean over streams.

    Args:
        d_model: residual stream width (must be divisible by n_hyper).
        n_hyper: number of parallel hyper-connection streams (default 2).
        n_sinkhorn_iters: Sinkhorn-Knopp iterations (paper uses 3-20).
    """

    def __init__(self, d_model: int, n_hyper: int = 2, n_sinkhorn_iters: int = 3):
        super().__init__()
        if d_model % n_hyper != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_hyper ({n_hyper})")
        self.d_model = d_model
        self.n_hyper = n_hyper
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.d_head = d_model // n_hyper

        self.q_proj = BitLinear_a4_8(d_model, d_model)
        self.k_proj = BitLinear_a4_8(d_model, self.d_head)

        identity_bias = torch.zeros(n_hyper, n_hyper)
        for i in range(n_hyper):
            identity_bias[i, i] = 5.0
        self.register_buffer("identity_bias", identity_bias)

        self.temperature = nn.Parameter(torch.ones(1))

        self.norm = RMSNorm(d_model)

    def sinkhorn_knopp(self, M: torch.Tensor, n_iters: int | None = None) -> torch.Tensor:
        """Project M onto the Birkhoff polytope (doubly-stochastic matrices).

        M: [..., n, n] real-valued matrix.
        Returns: [..., n, n] doubly-stochastic matrix (rows AND cols sum to 1).
        """
        if n_iters is None:
            n_iters = self.n_sinkhorn_iters

        M = M * self.temperature
        M = torch.exp(M - M.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0])

        for _ in range(n_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M

    def forward(self, current_x: torch.Tensor, streams: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Mix n_hyper streams using input-dependent doubly-stochastic H.

        Args:
            current_x: [B, T, d_model] — input to current layer.
            streams: tuple of n_hyper tensors, each [B, T, d_model].

        Returns:
            y: [B, T, d_model] — mixed stream (mean over n_hyper).
        """
        n = self.n_hyper
        if len(streams) != n:
            raise ValueError(f"Expected {n} streams, got {len(streams)}")
        B, T, _ = current_x.shape

        q = self.q_proj(current_x).view(B, T, n, self.d_head)

        ks = [self.k_proj(s) for s in streams]
        k_stack = torch.stack(ks, dim=2)

        H_logits = torch.einsum('btqd,btkd->btqk', q, k_stack) / math.sqrt(self.d_head)
        H_logits = H_logits + self.identity_bias

        H = self.sinkhorn_knopp(H_logits)

        streams_stack = torch.stack(streams, dim=2)
        y_streams = torch.einsum('btij,btjd->btid', H, streams_stack)
        y = y_streams.mean(dim=2)
        return self.norm(y)


class buselDecoderLayer(nn.Module):
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
        if self.mod_router.capacity_factor >= 1.0:
            attn_out = self.attn(self.attn_norm(x))
            moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
            return moe_out, aux_loss
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


class buselMTP4Pipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_weight = nn.Parameter(torch.randn(config.vocab_size, config.d_model) * 0.02)
        self.projections = nn.ModuleList([BitLinear_a4_8(config.d_model, config.d_model) for _ in range(3)])
        self.heads = nn.ModuleList([BitLinear_a4_8(config.d_model, config.vocab_size) for _ in range(4)])

    def _embed_lookup(self, token_ids):
        return self.embed_weight[token_ids.to(self.embed_weight.device)]

    def forward(self, main_hidden_states, next_token_ids=None):
        logits_t1 = self.heads[0](main_hidden_states)
        if next_token_ids is None or any(t is None for t in next_token_ids):
            return logits_t1, None, None, None
        h_detached = main_hidden_states.detach()
        combined_t2 = self.projections[0](h_detached) + self._embed_lookup(next_token_ids[0])
        logits_t2 = self.heads[1](combined_t2)
        combined_t3 = self.projections[1](combined_t2) + self._embed_lookup(next_token_ids[1])
        logits_t3 = self.heads[2](combined_t3)
        combined_t4 = self.projections[2](combined_t3) + self._embed_lookup(next_token_ids[2])
        logits_t4 = self.heads[3](combined_t4)
        return logits_t1, logits_t2, logits_t3, logits_t4


class buselModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hyper = int(getattr(config, "n_hyper", 2))

        capacity = 1.0
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            is_global = (l + 1) % 4 == 0
            self.layers.append(buselDecoderLayer(
                config.d_model, config.n_heads, config.expert_hidden,
                config.num_experts, is_global=is_global, capacity_factor=capacity
            ))

        self.m_residuals = nn.ModuleList([
            ManifoldConstrainedAttnRes(config.d_model, n_hyper=self.n_hyper)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.mtp_pipeline = buselMTP4Pipeline(config)
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self): self.use_gradient_checkpointing = True
    def disable_gradient_checkpointing(self): self.use_gradient_checkpointing = False

    def forward(self, x, next_token_ids=None, progress=0.0):
        nvtx_range_push("buselModel_Forward")
        streams = [x] * self.n_hyper
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            x = self.m_residuals[i](x, streams)

            if self.training and self.use_gradient_checkpointing and x.device.type in ["cuda", "mps"]:
                layer_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, x, progress, use_reentrant=False, determinism_check="none"
                )
            else:
                layer_out, aux_loss = layer(x, progress=progress)

            total_aux_loss += aux_loss
            streams = list(streams[1:]) + [layer_out]

        final_hidden = self.final_norm(x)
        mtp_outputs = self.mtp_pipeline(final_hidden, next_token_ids)
        nvtx_range_pop()
        return mtp_outputs, total_aux_loss

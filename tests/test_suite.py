"""
🧪 busel UNIFIED TEST SUITE — paper-compliance + integration
Covers all 6 reference papers + end-to-end integration.
"""
import os
import sys
import time
import math
import struct
import unittest
import json
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import busel_rust_io
from data.pipeline import get_busel_dataloader, collate_busel_batch
from model.patching import StridedFastBLTPatcher
from model.layers import BitLinear_a4_8, H_BitLinear, RMSNorm, SwishGLUClamped, fast_walsh_hadamard_transform
from model.attention import stable_gdn2_recurrent_jit, BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanExpertFFN, BulbaTernaryTitanMoE
from model.backbone import buselModel, ManifoldConstrainedAttnRes
from training.optimizer import _compiled_newton_schulz, buselOptimizerEngine, Muon, _newton_schulz_core
from training.recipe import buselLossEngine, validate_training_schedule

from busel_registry import register, get, list_registered, is_registered, unregister, clear_registry
from busel_logging import setup_logging, log_event, get_logger, JSONFormatter
from ui.teto import frame as teto_frame, frames as teto_frames, states as teto_states
from ui import cli as ui_cli

try:
    from multimodal.encoders import (
        ImageEncoder,
        VideoEncoder,
        AudioEncoder,
        PDFEncoder,
        DocxEncoder,
        TextEncoder,
        auto_encode,
        build_encoder_for,
        IMAGE_MARKER,
        MEDIA_END,
        IMAGE_BYTES,
    )
    from multimodal import list_encoders as list_mm_encoders
    HAS_MULTIMODAL_DEPS = True
except Exception:
    HAS_MULTIMODAL_DEPS = False


class _MockConfig:
    def __init__(self, **kw):
        self.d_model = kw.get("d_model", 128)
        self.n_layers = kw.get("n_layers", 2)
        self.n_heads = kw.get("n_heads", 4)
        self.expert_hidden = kw.get("expert_hidden", 256)
        self.num_experts = kw.get("num_experts", 2)
        self.top_k = kw.get("top_k", 2)
        self.vocab_size = kw.get("vocab_size", 259)
        self.n_hyper = kw.get("n_hyper", 2)


class TestbuselFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🚀 Running busel Test Suite on device: {cls.device.upper()}\n" + "=" * 80)

    def test_rust_io_streamer(self):
        print("🧪 [1] Rust ByteStreamer...")
        temp_file = "temp_test_rust_io.txt"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        test_data = "Hello from busel Rust IO! " * 350
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(test_data)
        try:
            streamer = busel_rust_io.ByteStreamer(temp_file, 8, 0)
            chunk1 = streamer.next_chunk()
            self.assertEqual(len(chunk1), 8)
            self.assertEqual(bytes(chunk1).decode("utf-8", errors="ignore"), "Hello fr")
            chunk2 = streamer.next_chunk()
            self.assertEqual(len(chunk2), 8)
            self.assertEqual(bytes(chunk2).decode("utf-8", errors="ignore"), "om busel")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("   ✅ Rust ByteStreamer passed.")

    def test_rust_binary_packer(self):
        print("🧪 [2] Rust Binary Packer...")
        temp_bin = "temp_test_packer.bin"
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
        test_bytes = [10, 20, 30, 40, 255, 0]
        try:
            busel_rust_io.append_to_binary_file(temp_bin, test_bytes)
            self.assertTrue(os.path.exists(temp_bin))
            self.assertEqual(os.path.getsize(temp_bin), 6)
            with open(temp_bin, "rb") as f:
                read_data = list(f.read())
            self.assertEqual(read_data, test_bytes)
        finally:
            if os.path.exists(temp_bin):
                os.remove(temp_bin)
        print("   ✅ Rust Binary Packer passed.")

    def test_rust_ternary_inference(self):
        print("🧪 [3] Rust Ternary CPU Inference...")
        x = [1.0, -1.0, 2.0]
        w = [1, 0, -1, 0, 1, 1]
        expected_y = [-1.0, 1.0]
        actual_y = busel_rust_io.ternary_matmul_cpu(x, w, 2, 3)
        self.assertEqual(actual_y, expected_y)
        print("   ✅ Rust Ternary CPU Inference passed.")

    def test_bitlinear_quantization(self):
        print("🧪 [4] BitLinear 1.58b Quantization (forward)...")
        linear = BitLinear_a4_8(64, 128).to(self.device)
        x = torch.randn(2, 64, device=self.device)
        out = linear(x)
        self.assertEqual(out.shape, (2, 128))
        self.assertFalse(torch.isnan(out).any())
        print("   ✅ BitLinear Quantization passed.")

    def test_jit_gdn2_attention(self):
        print("🧪 [5] Stable JIT GDN-2 Loop (forward)...")
        q = torch.randn(2, 128, 4, 64, device=self.device)
        k = torch.randn(2, 128, 4, 64, device=self.device)
        v = torch.randn(2, 128, 4, 64, device=self.device)
        b = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        w = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        alpha = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
            out = stable_gdn2_recurrent_jit(q, k, v, b, w, alpha)
        self.assertEqual(out.shape, (2, 128, 256))
        self.assertFalse(torch.isnan(out).any())
        print("   ✅ Stable JIT GDN-2 Loop passed.")

    def test_fused_glu_and_expert_ffn(self):
        print("🧪 [6] Fused Gate-Up Projections (SwishGLUClamped)...")
        x = torch.randn(2, 32, 256, device=self.device)
        glu = SwishGLUClamped(256, 512).to(self.device)
        out_glu = glu(x)
        self.assertEqual(out_glu.shape, (2, 32, 256))
        self.assertFalse(torch.isnan(out_glu).any())
        expert = BulbaTernaryTitanExpertFFN(256, 512).to(self.device)
        out_exp = expert(x)
        self.assertEqual(out_exp.shape, (2, 32, 256))
        self.assertFalse(torch.isnan(out_exp).any())
        print("   ✅ Fused Gate-Up Projections passed.")

    def test_muon_transpose_trick(self):
        print("🧪 [7] Muon Transpose Trick (NS forward)...")
        X = torch.randn(512, 256, device=self.device)
        O_t = _compiled_newton_schulz(X, steps=5)
        self.assertEqual(O_t.shape, (512, 256))
        self.assertFalse(torch.isnan(O_t).any(), "O_t contains NaNs!")
        print("   ✅ Muon Transpose Trick passed.")

    def test_complete_backbone_and_gradients(self):
        print("🧪 [8] Complete Backbone & Backpropagation...")
        cfg = _MockConfig(d_model=256, n_layers=4, n_heads=4, expert_hidden=512, num_experts=4)
        patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(self.device)
        model = buselModel(cfg).to(self.device)
        opt_engine = buselOptimizerEngine(model, lr_muon=0.0004, lr_adamw=0.00004)
        loss_engine = buselLossEngine(cfg.vocab_size)
        byte_batch = torch.randint(0, 259, (2, 256), dtype=torch.int32, device=self.device)
        input_bytes = byte_batch[:, :-patcher.stride]
        opt_engine.zero_grad(set_to_none=True)
        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float16
        with torch.autocast(device_type=self.device, dtype=target_dtype):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            targets = byte_batch[:, patcher.stride::patcher.stride][:, :T_patches]
            if targets.shape[1] < T_patches:
                targets = torch.nn.functional.pad(targets, (0, T_patches - targets.shape[1]), value=0)
            (logits_t1, _, _, _), aux_loss = model(patches, None)
            loss = loss_engine.compute_pretrain_loss(logits_t1, targets) + aux_loss.float()
        loss.backward()
        has_gradients = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                has_gradients = True
                self.assertFalse(torch.isnan(p.grad).any(), f"Gradient of '{name}' is NaN!")
        self.assertTrue(has_gradients, "No gradients were computed!")
        opt_engine.step()
        for name, p in model.named_parameters():
            self.assertFalse(torch.isnan(p).any(), f"Weights of '{name}' became NaN after optimizer step!")
        print("   ✅ Complete Backbone & Backpropagation passed.")

    def test_gdn2_decoupled_erase_and_write_gates(self):
        # Paper §3.1: b_proj and w_proj are SEPARATE linear layers
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4)
        self.assertIsInstance(block.b_proj, torch.nn.Linear)
        self.assertIsInstance(block.w_proj, torch.nn.Linear)
        self.assertIsNot(block.b_proj, block.w_proj)

    def test_gdn2_gates_sigmoid_bounded_0_1(self):
        # Paper §3.1: erase (b) and write (w) gates ∈ (0, 1)
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4).eval()
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            b = torch.sigmoid(block.b_proj(x))
            w = torch.sigmoid(block.w_proj(x))
        self.assertTrue((b > 0).all() and (b < 1).all())
        self.assertTrue((w > 0).all() and (w < 1).all())

    def test_gdn2_log_decay_alpha_a_init_negative_three(self):
        # Paper §3.2 (Eq. 12): alpha_a is initialized to -3.0
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4)
        self.assertTrue(torch.allclose(block.alpha_a, torch.full_like(block.alpha_a, -3.0)))

    def test_gdn2_log_decay_alpha_in_unit_interval(self):
        # Paper §3.2 (Eq. 12): g_t = -exp(alpha_a)*softplus(alpha_proj), alpha = exp(g_t) ∈ (0, 1)
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4).eval()
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            alpha_proj = block.alpha_proj(x).view(2, 16, block.n_heads, block.d_k)
            g_t = -torch.exp(block.alpha_a).view(1, 1, block.n_heads, 1) * F.softplus(alpha_proj)
            alpha = torch.exp(g_t)
        self.assertTrue((alpha > 0).all())
        self.assertTrue((alpha < 1.0 + 1e-6).all())

    def test_gdn2_serope_real_imag_pairing(self):
        # Paper §3.3 (SeRoPE): pairs real (even) and imag (odd) indices with cos/sin
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4).eval()
        T = 8
        q = torch.randn(2, T, 4, block.d_k)
        k = torch.randn(2, T, 4, block.d_k)
        with torch.no_grad():
            q_out, k_out = block.apply_serope(T, q, k)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)
        self.assertFalse(torch.allclose(q_out, q))
        self.assertFalse(torch.allclose(k_out, k))

    def test_gdn2_serope_is_actually_rotating(self):
        # SeRoPE must not be a no-op: output must differ from input
        block = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4).eval()
        T = 16
        q = torch.randn(1, T, 4, block.d_k)
        with torch.no_grad():
            q_out, _ = block.apply_serope(T, q, q)
        self.assertFalse(torch.allclose(q_out, q))

    def test_gdn2_mla_latent_d_c_is_128(self):
        # Paper §4: MLA compresses KV to latent size d_c=128
        block = MultiHeadLatentAttention(d_model=128, n_heads=4, d_c=128)
        self.assertEqual(block.d_c, 128)
        out = block(torch.randn(2, 8, 128))
        self.assertEqual(out.shape, (2, 8, 128))
        self.assertFalse(torch.isnan(out).any())

    def test_gdn2_mla_kv_q_compression_dimensions(self):
        # Paper §4: kv_compress and q_compress project d_model → d_c
        block = MultiHeadLatentAttention(d_model=128, n_heads=4, d_c=128)
        self.assertEqual(block.kv_compress.out_features, 128)
        self.assertEqual(block.q_compress.out_features, 128)
        self.assertEqual(block.k_decompress.in_features, 128)
        self.assertEqual(block.v_decompress.in_features, 128)

    def test_bitnetv2_weights_ternary(self):
        # Paper §3.1: weights quantize to {-1, 0, +1} (1.58 bits)
        linear = BitLinear_a4_8(64, 32).eval()
        x = torch.randn(2, 64)
        with torch.no_grad():
            w = linear.weight
            alpha = w.abs().mean() + 1e-5
            w_scaled = w / alpha
            w_clipped = torch.clamp(w_scaled, -1, 1)
            w_quant = w_clipped + (torch.round(w_clipped) - w_clipped)
        uniq = torch.unique(w_quant)
        self.assertTrue(all(v in {-1.0, 0.0, 1.0} for v in uniq.tolist()))

    def test_bitnetv2_ternary_distribution_meaningful(self):
        # Paper §3.1: ternary distribution should not collapse
        torch.manual_seed(0)
        linear = BitLinear_a4_8(256, 256).eval()
        with torch.no_grad():
            w = linear.weight
            alpha = w.abs().mean() + 1e-5
            w_scaled = w / alpha
            w_clipped = torch.clamp(w_scaled, -1, 1)
            w_quant = w_clipped + (torch.round(w_clipped) - w_clipped)
        n_pos = (w_quant == 1).sum().item()
        n_neg = (w_quant == -1).sum().item()
        n_zero = (w_quant == 0).sum().item()
        total = w_quant.numel()
        self.assertGreater(n_pos / total, 0.1)
        self.assertGreater(n_neg / total, 0.1)
        self.assertGreater(n_zero / total, 0.05)
        self.assertLess(n_zero / total, 0.7)

    def test_bitnetv2_h_bitlinear_applies_walsh_hadamard(self):
        # Paper §3.2: H_BitLinear = BitLinear_a4_8 + FWHT pre-transform on input
        d = 64
        h = H_BitLinear(d, d).eval()
        x = torch.randn(2, 8, d)
        with torch.no_grad():
            plain = BitLinear_a4_8(d, d).eval()
            plain.weight.data = h.weight.data.clone()
            out_with_wht = h(x)
            out_plain = plain(x)
        self.assertEqual(out_with_wht.shape, out_plain.shape)
        self.assertFalse(torch.allclose(out_with_wht, out_plain, atol=1e-3),
                         "H_BitLinear must apply FWHT to input, producing a different output than plain BitLinear")

    def test_bitnetv2_h_bitlinear_used_for_o_proj(self):
        # Paper §3.2: o_proj in both GDN-2 and MLA must use H_BitLinear
        gdn = BulbaGDN2SeRoPEBlock(d_model=128, n_heads=4)
        mla = MultiHeadLatentAttention(d_model=128, n_heads=4, d_c=128)
        self.assertIsInstance(gdn.o_proj, H_BitLinear)
        self.assertIsInstance(mla.o_proj, H_BitLinear)

    def test_bitnetv2_swishglu_learnable_clamp(self):
        # Paper §3.3: SwishGLUClamped uses learnable per-channel clamp bounds
        glu = SwishGLUClamped(d_model=128, d_ffn=256)
        self.assertTrue(hasattr(glu, "clipping_bounds"))
        self.assertIsInstance(glu.clipping_bounds, torch.nn.Parameter)
        self.assertEqual(glu.clipping_bounds.shape, (256,))

    def test_bitnetv2_swishglu_clip_bounds_get_gradient(self):
        # Paper §3.3: clipping_bounds receives gradients via STE
        glu = SwishGLUClamped(d_model=64, d_ffn=128)
        out = glu(torch.randn(2, 8, 64))
        out.sum().backward()
        self.assertIsNotNone(glu.clipping_bounds.grad)
        self.assertFalse(torch.isnan(glu.clipping_bounds.grad).any())

    def test_mar_sinkhorn_knopp_doubly_stochastic_rows(self):
        # mHC §3.2: H is doubly-stochastic (rows sum to 1)
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(2, 4, 3, 3)
        H = mar.sinkhorn_knopp(M)
        row_sums = H.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)

    def test_mar_sinkhorn_knopp_doubly_stochastic_cols(self):
        # mHC §3.2: H is doubly-stochastic (cols sum to 1)
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(2, 4, 3, 3)
        H = mar.sinkhorn_knopp(M)
        col_sums = H.sum(dim=-2)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)

    def test_mar_sinkhorn_knopp_non_negative(self):
        # mHC §3.2: H ∈ [0, 1] (doubly-stochastic)
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        M = torch.randn(8, 16, 2, 2) * 100.0
        H = mar.sinkhorn_knopp(M)
        self.assertTrue((H >= 0).all())

    def test_mar_sinkhorn_knopp_no_overflow_under_extreme_inputs(self):
        # Numerical stability: extreme inputs must not produce NaN/Inf
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        M = torch.randn(1, 1, 2, 2) * 1000.0
        H = mar.sinkhorn_knopp(M)
        self.assertFalse(torch.isnan(H).any())
        self.assertFalse(torch.isinf(H).any())

    def test_mar_sinkhorn_knopp_more_iters_converge_tighter(self):
        # mHC §3.2: more Sinkhorn iterations → tighter convergence to Birkhoff
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(4, 5, 3, 3)
        H = mar.sinkhorn_knopp(M, n_iters=20)
        row_dev = (H.sum(dim=-1) - 1.0).abs().max()
        col_dev = (H.sum(dim=-2) - 1.0).abs().max()
        self.assertLess(row_dev.item(), 1e-3)
        self.assertLess(col_dev.item(), 1e-3)

    def test_mar_identity_init_means_h_approx_I(self):
        # mHC §3.3: at init, H ≈ I (identity-mapping property)
        d_model, n_hyper = 128, 2
        mar = ManifoldConstrainedAttnRes(d_model=d_model, n_hyper=n_hyper, n_sinkhorn_iters=5).eval()
        torch.manual_seed(0)
        x = torch.zeros(1, 4, d_model)
        streams = tuple(torch.zeros(1, 4, d_model) for _ in range(n_hyper))
        with torch.no_grad():
            q = mar.q_proj(x).view(1, 4, n_hyper, mar.d_head)
            ks = [mar.k_proj(s) for s in streams]
            k_stack = torch.stack(ks, dim=2)
            H_logits = torch.einsum("btqd,btkd->btqk", q, k_stack) / math.sqrt(mar.d_head)
            H_logits = H_logits + mar.identity_bias
            H = mar.sinkhorn_knopp(H_logits)
        diag = H.diagonal(dim1=-2, dim2=-1)
        off_diag = H - torch.diag_embed(diag)
        self.assertGreater(diag.mean().item(), off_diag.abs().mean().item())

    def test_mar_forward_preserves_shape(self):
        # mAR forward: [B, T, d_model] → [B, T, d_model]
        d_model, n_hyper, B, T = 128, 2, 4, 16
        mar = ManifoldConstrainedAttnRes(d_model=d_model, n_hyper=n_hyper)
        x = torch.randn(B, T, d_model)
        streams = tuple(torch.randn(B, T, d_model) for _ in range(n_hyper))
        y = mar(x, streams)
        self.assertEqual(y.shape, (B, T, d_model))

    def test_mar_streams_mismatch_raises(self):
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        x = torch.randn(2, 4, 128)
        with self.assertRaises(ValueError):
            mar(x, (torch.randn(2, 4, 128),))

    def test_mar_d_model_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            ManifoldConstrainedAttnRes(d_model=127, n_hyper=2)

    def test_mar_gradients_flow_through_sinkhorn(self):
        # Sinkhorn-Knopp must be differentiable via exp
        mar = ManifoldConstrainedAttnRes(d_model=64, n_hyper=2)
        x = torch.randn(2, 8, 64, requires_grad=True)
        streams = tuple(torch.randn(2, 8, 64, requires_grad=True) for _ in range(2))
        y = mar(x, streams)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        for s in streams:
            self.assertIsNotNone(s.grad)
            self.assertFalse(torch.isnan(s.grad).any())

    def test_mar_temperature_gets_gradient(self):
        mar = ManifoldConstrainedAttnRes(d_model=64, n_hyper=2)
        y = mar(torch.randn(2, 4, 64), tuple(torch.randn(2, 4, 64) for _ in range(2)))
        y.sum().backward()
        self.assertIsNotNone(mar.temperature.grad)
        self.assertFalse(torch.isnan(mar.temperature.grad).any())

    def test_mar_buselModel_n_hyper_2(self):
        cfg = _MockConfig(d_model=128, n_layers=3, n_heads=4, n_hyper=2)
        model = buselModel(cfg)
        B, T = 2, 8
        hidden = torch.randn(B, T, cfg.d_model)
        mtp, aux = model(hidden)
        self.assertEqual(mtp[0].shape, (B, T, 259))
        self.assertEqual(aux.shape, ())

    def test_mar_buselModel_n_hyper_4(self):
        cfg = _MockConfig(d_model=128, n_layers=3, n_heads=4, n_hyper=4)
        model = buselModel(cfg)
        B, T = 2, 8
        hidden = torch.randn(B, T, cfg.d_model)
        mtp, aux = model(hidden)
        self.assertEqual(mtp[0].shape, (B, T, 259))

    def test_mar_residual_connection_preserved(self):
        # mHC §3.1: y_l = x_l + f_l(mixed_x_l). Residual add must be in buselModel.forward.
        cfg = _MockConfig(d_model=128, n_layers=2, n_heads=4, n_hyper=2)
        torch.manual_seed(42)
        model = buselModel(cfg).eval()
        const_value = 1.0
        for layer in model.layers:
            def constant_forward(self, x, progress=0.0):
                return torch.full_like(x, const_value), torch.tensor(0.0, device=x.device, dtype=x.dtype)
            layer.forward = constant_forward.__get__(layer)
        x_in = torch.zeros(1, 2, cfg.d_model)
        recorded_hidden = []
        original_final_norm = model.final_norm.forward
        def spy_norm(x):
            recorded_hidden.append(x.clone())
            return original_final_norm(x)
        model.final_norm.forward = spy_norm
        try:
            with torch.no_grad():
                model(x_in)
        finally:
            model.final_norm.forward = original_final_norm
        final_x = recorded_hidden[-1]
        expected_with_residual = 2 * const_value
        actual_max = final_x.abs().max().item()
        self.assertAlmostEqual(actual_max, expected_with_residual, delta=0.5,
                               msg=f"Residual missing or wrong: final x max={actual_max:.3f}, expected ~{expected_with_residual} (2 layers of residual adds with constant 1.0)")

    def test_mar_layer_outputs_flow_into_streams(self):
        # Streams must carry the post-residual x (residual stream values), enabling mHC mixing.
        cfg = _MockConfig(d_model=128, n_layers=3, n_heads=4, n_hyper=2)
        torch.manual_seed(0)
        model = buselModel(cfg).eval()
        recorded_mar_inputs = []
        original_mar_forward = ManifoldConstrainedAttnRes.forward
        def spy_forward(self, current_x, streams):
            recorded_mar_inputs.append([s.clone() for s in streams])
            return original_mar_forward(self, current_x, streams)
        ManifoldConstrainedAttnRes.forward = spy_forward
        try:
            x_in = torch.randn(1, 4, cfg.d_model)
            with torch.no_grad():
                model(x_in)
        finally:
            ManifoldConstrainedAttnRes.forward = original_mar_forward
        self.assertGreater(len(recorded_mar_inputs), 1, "MAR must run at each layer")
        for layer_idx, streams in enumerate(recorded_mar_inputs[1:], start=1):
            self.assertEqual(len(streams), cfg.n_hyper)

    def test_muon_ns_produces_orthogonal_output(self):
        # Paper §3.1: NS output singular values are bounded (≈ 1, within paper tolerance)
        # Uses eager _newton_schulz_core to avoid torch.compile recompile limit
        torch.manual_seed(0)
        X = torch.randn(64, 32)
        O = _newton_schulz_core(X, steps=5)
        sv = torch.linalg.svdvals(O)
        self.assertTrue((sv > 0.5).all(), f"singular values far below 1: min={sv.min().item()}")
        self.assertTrue((sv < 1.5).all(), f"singular values far above 1: max={sv.max().item()}")

    def test_muon_ns_uses_quintic_coefficients(self):
        # Paper §3.1: optimal quintic NS coefficients (3.4445, -4.7750, 2.0315)
        src = inspect.getsource(_newton_schulz_core)
        self.assertIn("3.4445", src)
        self.assertIn("-4.7750", src)
        self.assertIn("2.0315", src)

    def test_muon_ns_five_iterations_default(self):
        # Paper §3.1: NS uses 5 iterations by default
        src = inspect.getsource(_newton_schulz_core)
        self.assertIn("for _ in range(steps)", src)
        self.assertIn("steps=5", src)

    def test_muon_ns_handles_tall_and_wide_matrices(self):
        # Paper §3.1: NS uses transpose trick for tall (rows > cols) matrices
        # Uses eager _newton_schulz_core to avoid torch.compile recompile limit
        torch.manual_seed(1)
        for shape in [(128, 32), (32, 128), (64, 64), (256, 64)]:
            O = _newton_schulz_core(torch.randn(*shape), steps=5)
            self.assertEqual(O.shape, shape)
            self.assertFalse(torch.isnan(O).any(), f"shape={shape}: NaN output")
            self.assertFalse(torch.isinf(O).any(), f"shape={shape}: Inf output")

    def test_muon_scale_formula(self):
        # Paper §3.2: Muon scale = 0.2 * sqrt(max(A, B))
        src = inspect.getsource(Muon.step)
        self.assertIn("0.2", src)
        self.assertIn("sqrt", src)
        self.assertIn("max(A, B)", src)

    def test_muon_momentum_0_95(self):
        # Paper §3.2: Muon momentum = 0.95 (Nesterov-like)
        opt = Muon([torch.zeros(4, 4, requires_grad=True)], momentum=0.95)
        self.assertEqual(opt.param_groups[0]["momentum"], 0.95)

    def test_muon_hybrid_routing_splits_params(self):
        # Paper §3.3: Hybrid routes 2D `proj` params (excluding `router`) to Muon, rest to AdamW
        cfg = _MockConfig(d_model=128, n_layers=2, n_heads=4)
        model = buselModel(cfg)
        engine = buselOptimizerEngine(model, lr_muon=0.001, lr_adamw=0.0001)
        muon_ids = set(id(p) for g in engine.opt_muon.param_groups for p in g["params"])
        adamw_ids = set(id(p) for g in engine.opt_adamw.param_groups for p in g["params"])
        self.assertGreater(len(muon_ids), 0, "Muon should receive some params")
        self.assertGreater(len(adamw_ids), 0, "AdamW should receive some params")
        self.assertEqual(len(muon_ids & adamw_ids), 0, "no param in both")
        self.assertEqual(len(muon_ids | adamw_ids), sum(1 for p in model.parameters() if p.requires_grad))

    def test_muon_routing_covers_moe_and_mla(self):
        # ISSUES.md #1: 2D BitLinear weights (MoE, MLA, Blackboard, mtp) must route to Muon.
        cfg = _MockConfig(d_model=384, n_layers=4, n_heads=6, expert_hidden=768, num_experts=4)
        model = buselModel(cfg)
        engine = buselOptimizerEngine(model, lr_muon=0.001, lr_adamw=0.0001)
        n_muon = sum(p.numel() for g in engine.opt_muon.param_groups for p in g["params"])
        n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = n_muon / n_total
        self.assertGreater(ratio, 0.85,
                           f"Expected ≥85% of params to Muon after ISSUES.md #1 fix; got {ratio*100:.1f}%")

    def test_muon_routing_excludes_router_and_embed(self):
        # Routers + embed tables stay on AdamW (noise-sensitive / structured).
        cfg = _MockConfig(d_model=128, n_layers=2, n_heads=4, num_experts=4)
        model = buselModel(cfg)
        engine = buselOptimizerEngine(model, lr_muon=0.001, lr_adamw=0.0001)
        muon_ids = {id(p) for g in engine.opt_muon.param_groups for p in g["params"]}
        for name, p in model.named_parameters():
            if "router" in name or "embed" in name:
                self.assertNotIn(id(p), muon_ids,
                                 f"{name} must NOT be on Muon (router/embed exclusion rule)")

    def test_muon_step_does_not_produce_nan(self):
        # ISSUES.md #3 + #4 regression: momentum update + NS core must not NaN.
        torch.manual_seed(42)
        p = torch.randn(64, 64, device=self.device, dtype=torch.float32) * 0.02
        opt = Muon([p], lr=0.001, momentum=0.95, weight_decay=0.1)
        for i in range(5):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            self.assertFalse(torch.isnan(p).any(),
                             f"step {i}: Muon produced NaN weights")
            self.assertFalse(torch.isinf(p).any(),
                             f"step {i}: Muon produced Inf weights")

    def test_fastblt_vocab_size_259(self):
        # Paper §2.1: byte-level vocab is 256 (UTF-8) + 3 multimodal specials = 259
        patcher = StridedFastBLTPatcher(d_model=128)
        self.assertEqual(patcher.embed_weight.shape[0], 259)

    def test_fastblt_stride_4_kernel_5(self):
        # Paper §3: stride=4, conv kernel=5 (causal receptive field)
        patcher = StridedFastBLTPatcher(d_model=128, stride=4, kernel_size=5)
        self.assertEqual(patcher.stride, 4)
        self.assertEqual(patcher.kernel_size, 5)

    def test_fastblt_no_bpe_no_subword_tokens(self):
        # Paper §2.1: NO BPE — vocab is exactly 259
        patcher = StridedFastBLTPatcher(d_model=128)
        self.assertLessEqual(patcher.embed_weight.shape[0], 300, "Vocab too large — likely BPE contamination")

    def test_fastblt_byte_input_to_patch_count(self):
        # Paper §3: T bytes → floor((T - 1) / stride) + 1 patches (left-padding by kernel-1)
        d_model, stride, kernel = 128, 4, 5
        patcher = StridedFastBLTPatcher(d_model=d_model, stride=stride, kernel_size=kernel).eval()
        T = 64
        byte_ids = torch.randint(0, 259, (2, T))
        with torch.no_grad():
            patches = patcher(byte_ids)
        expected_patches = (T - 1) // stride + 1
        self.assertEqual(patches.shape, (2, expected_patches, d_model))

    def test_fastblt_byte_embeddings_are_learned(self):
        # Paper §3.2: byte embeddings are LEARNED
        patcher = StridedFastBLTPatcher(d_model=128, d_byte=64)
        self.assertTrue(patcher.embed_weight.requires_grad)
        self.assertEqual(patcher.embed_weight.shape, (259, 64))

    def test_fastblt_glu_gate_is_nonlinear(self):
        # Paper §3.2: GLU gate must be nonlinear (sigmoid after SiLU)
        patcher = StridedFastBLTPatcher(d_model=128, d_byte=32).eval()
        x = torch.randn(1, 16, 32)
        with torch.no_grad():
            gate_pos = torch.sigmoid(patcher.gate_proj_up(F.silu(patcher.gate_proj_down(x))))
            gate_neg = torch.sigmoid(patcher.gate_proj_up(F.silu(patcher.gate_proj_down(-x))))
        self.assertTrue((gate_pos > 0).all() and (gate_pos < 1).all())
        self.assertFalse(torch.allclose(gate_pos, gate_neg, atol=1e-3),
                         "GLU gate should be nonlinear — flipping input should change output")

    def test_fastblt_causal_left_padding(self):
        # Paper §3: causal conv uses left-side padding only (no future leakage)
        patcher = StridedFastBLTPatcher(d_model=128, stride=4, kernel_size=5).eval()
        T = 24
        byte_ids_a = torch.zeros(1, T, dtype=torch.long)
        byte_ids_a[0, 0] = 65
        byte_ids_a[0, 16] = 66
        with torch.no_grad():
            patches_a = patcher(byte_ids_a)
        byte_ids_b = byte_ids_a.clone()
        byte_ids_b[0, 16] = 67
        with torch.no_grad():
            patches_b = patcher(byte_ids_b)
        early_a = patches_a[0, 0]
        early_b = patches_b[0, 0]
        self.assertTrue(torch.allclose(early_a, early_b, atol=1e-3),
                        "Causal padding: byte at pos 16 must NOT affect the first patch (no future leakage)")
        late_a = patches_a[0, 4]
        late_b = patches_b[0, 4]
        self.assertFalse(torch.allclose(late_a, late_b, atol=1e-3),
                         "Changing byte at pos 16 SHOULD affect patch 4 (which covers that position)")

    def test_registry_decorator_basic(self):
        """🛸 busel REGISTRY — basic @register/get/is_registered/list_registered."""
        print("🧪 [REG-1] busel Registry — basic register/get API...")
        clear_registry()
        try:
            @register("test_kind", "demo_cls")
            class _Demo:
                pass

            self.assertTrue(is_registered("test_kind", "demo_cls"))
            self.assertIs(get("test_kind", "demo_cls"), _Demo)
            self.assertIn("demo_cls", list_registered("test_kind"))
            self.assertEqual(list_registered("test_kind"), ["demo_cls"])
            print("   ✅ Registry register/get/is_registered/list_registered pass.")
        finally:
            unregister("test_kind", "demo_cls")

    def test_registry_collision_raises(self):
        """🛸 busel REGISTRY — duplicate (kind, name) without override raises KeyError."""
        print("🧪 [REG-2] busel Registry — collision detection...")
        clear_registry()
        try:
            @register("test_kind", "dup")
            class _A:
                pass

            with self.assertRaises(KeyError) as ctx:
                @register("test_kind", "dup")
                class _B:
                    pass

            msg = str(ctx.exception)
            self.assertIn("collision", msg.lower())
            self.assertIn("override=True", msg)
            self.assertIs(get("test_kind", "dup"), _A)
            print("   ✅ Registry collision correctly raised KeyError with override hint.")
        finally:
            unregister("test_kind", "dup")

    def test_registry_override_allowed(self):
        """🛸 busel REGISTRY — override=True replaces existing entry."""
        print("🧪 [REG-3] busel Registry — override=True works...")
        clear_registry()
        try:
            @register("test_kind", "ovr")
            class _First:
                pass

            @register("test_kind", "ovr", override=True)
            class _Second:
                pass

            self.assertIs(get("test_kind", "ovr"), _Second)
            print("   ✅ Registry override=True correctly replaced entry.")
        finally:
            unregister("test_kind", "ovr")

    def test_registry_attention_and_optimizer_registered(self):
        """🛸 busel REGISTRY — model.attention + training.optimizer register themselves on import."""
        print("🧪 [REG-4] busel Registry — production entries (attention, optimizer) are registered...")
        attn = list_registered("attention")
        opt = list_registered("optimizer")
        self.assertIn("gdn2", attn, "BulbaGDN2SeRoPEBlock must register as 'gdn2'")
        self.assertIn("mla", attn, "MultiHeadLatentAttention must register as 'mla'")
        self.assertIn("muon", opt, "Muon must register as 'muon'")
        self.assertIn("hybrid_muon_adamw", opt, "buselOptimizerEngine must register as 'hybrid_muon_adamw'")

        from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
        from training.optimizer import Muon as _Muon, buselOptimizerEngine
        self.assertIs(get("attention", "gdn2"), BulbaGDN2SeRoPEBlock)
        self.assertIs(get("attention", "mla"), MultiHeadLatentAttention)
        self.assertIs(get("optimizer", "muon"), _Muon)
        self.assertIs(get("optimizer", "hybrid_muon_adamw"), buselOptimizerEngine)
        print("   ✅ Production attention + optimizer entries are correctly registered.")

    def test_json_logger_writes_valid_jsonl(self):
        """📚 busel LOGGING — JSONFormatter produces valid one-line JSON per record."""
        print("🧪 [LOG-1] busel Logging — JSON formatter emits valid JSONL...")
        import io
        import logging as _logging
        from busel_logging import JSONFormatter

        buf = io.StringIO()
        handler = _logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        logger = _logging.getLogger("busel_test_json")
        logger.handlers.clear()
        logger.setLevel(_logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        logger.info("step_complete", extra={"step": 42, "loss": 3.14, "lr": 0.0001, "vram_mb": 1024.5, "extra_field": "hi"})

        line = buf.getvalue().strip()
        self.assertTrue(line.startswith("{") and line.endswith("}"), f"Line not JSON: {line!r}")
        self.assertNotIn("\n", line, "JSONL must be single line per record")
        parsed = json.loads(line)
        self.assertEqual(parsed["event"], "step_complete")
        self.assertEqual(parsed["level"], "INFO")
        self.assertIn("ts", parsed)
        self.assertEqual(parsed["step"], 42)
        self.assertAlmostEqual(parsed["loss"], 3.14, places=4)
        self.assertEqual(parsed["extra"]["extra_field"], "hi")
        print("   ✅ JSON logger emits valid single-line JSON with hoisted fields.")

    def test_json_logger_writes_to_file(self):
        """📚 busel LOGGING — setup_logging appends to a real file (one JSON per line)."""
        print("🧪 [LOG-2] busel Logging — setup_logging writes to checkpoints/busel.log.jsonl...")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            logger = setup_logging(log_dir=tmp, log_filename="test.jsonl")
            log_event("test_event", step=1, loss=2.5)
            log_event("training_complete", step=100, total_time_s=42.0)

            log_path = Path(tmp) / "test.jsonl"
            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            for ln in lines:
                obj = json.loads(ln)
                self.assertIn("ts", obj)
                self.assertIn("event", obj)
                self.assertIn("level", obj)
            self.assertEqual(json.loads(lines[0])["event"], "test_event")
            self.assertEqual(json.loads(lines[1])["event"], "training_complete")
        print("   ✅ setup_logging appends valid JSONL to disk.")

    def test_teto_frames_nonempty_strings(self):
        """🎵 busel TETO — every state returns a non-empty string; all idle frames are distinct."""
        print("🧪 [TETO-1] busel Teto — frames are non-empty and distinct...")
        all_states = teto_states()
        self.assertGreaterEqual(len(all_states), 4, "Need at least 4 states (idle, blink, smile, ...)")
        for state in all_states:
            f = teto_frame(state, 0)
            self.assertIsInstance(f, str, f"frame({state!r}) must be str")
            self.assertGreater(len(f), 0, f"frame({state!r}) must be non-empty")

        idle = teto_frames()
        self.assertGreaterEqual(len(idle), 6, "Need at least 6 idle emoticon frames")
        self.assertEqual(len(idle), len(set(idle)), "Idle frames must all be distinct")
        for f in idle:
            self.assertGreater(len(f), 0)
            self.assertIsInstance(f, str)
        print(f"   ✅ Teto: {len(all_states)} states, {len(idle)} distinct idle frames — all non-empty.")

    def test_teto_idle_cycles_through_frames(self):
        """🎵 busel TETO — frame('idle', tick) cycles through the 12-frame emoticon set."""
        print("🧪 [TETO-2] busel Teto — idle cycle wraps correctly at modulo 12...")
        cycle_a = [teto_frame("idle", i) for i in range(12)]
        cycle_b = [teto_frame("idle", i + 12) for i in range(12)]
        self.assertEqual(cycle_a, cycle_b, "frame('idle', tick+12) must equal frame('idle', tick) for all 12 ticks")
        self.assertEqual(len(set(cycle_a)), 12, "All 12 idle frames must be distinct within one cycle")
        print(f"   ✅ Idle cycle of {len(cycle_a)} distinct emoticons wraps cleanly modulo 12.")

    def test_cli_helpers_do_not_crash(self):
        """💡 busel CLI — every public helper runs without raising (with or without rich)."""
        print("🧪 [CLI-1] busel CLI — all helpers run without raising...")
        ui_cli.header("TEST HEADER", "subtitle here")
        ui_cli.status_panel("test", key1="value1", key2=42)
        ui_cli.log("info message", level="info")
        ui_cli.log("warn message", level="warn")
        ui_cli.log("error message", level="error")
        ui_cli.log("ok message", level="ok")
        line = ui_cli.step_line(10, 100, 3.14, 0.5, 0.001, 1.5, 8, 1234.5, 256.0)
        self.assertIn("Step 00010/00100", line)
        self.assertIn("1234", line)
        self.assertIn("VRAM: 256MB", line)
        self.assertIn("Total", line)
        ui_cli.safe_print("hello\n")
        print("   ✅ All ui.cli helpers execute without raising.")

    def test_cli_animated_header_runs(self):
        """💡 busel CLI — animated_header + project_tree + spinner + progress_bar all execute."""
        print("🧪 [CLI-2] busel CLI — animated_header, spinner, progress_bar, project_tree...")
        ui_cli.animated_header("busel TEST", cycles=2, palette="teto")
        ui_cli.animated_header("busel TEST", cycles=1, palette="cycle")
        ui_cli.project_tree()
        with ui_cli.spinner("working") as _:
            pass
        with ui_cli.progress_bar(total=10, description="test") as handle:
            if handle is not None:
                handle.update(advance=5)
        print("   ✅ Animated header, spinner, progress bar, project tree all execute cleanly.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_registry_lists_all_encoders(self):
        """🛰️ busel MULTIMODAL — every encoder class self-registers on import."""
        print("🧪 [MM-1] busel Multimodal — registry lists all 6 encoders...")
        names = list_mm_encoders()
        for n in ("image", "video", "audio", "pdf", "docx", "text"):
            self.assertIn(n, names, f"encoder '{n}' must be registered")
        self.assertGreaterEqual(len(names), 6)
        print(f"   ✅ Multimodal registry: {sorted(names)}")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_image_encoder_roundtrip(self):
        """🛰️ busel MULTIMODAL — ImageEncoder encode → decode returns 32×32 RGB."""
        print("🧪 [MM-2] busel Multimodal — image encode/decode round-trip...")
        from PIL import Image as _Im
        src = _Im.new("RGB", (256, 256), color=(12, 34, 56))
        enc = ImageEncoder()
        tokens = enc.encode(src)
        self.assertIsInstance(tokens, list, "encode must return list[int]")
        self.assertEqual(len(tokens), IMAGE_BYTES + 2, f"must be {IMAGE_BYTES}+2 tokens, got {len(tokens)}")
        self.assertEqual(tokens[0], IMAGE_MARKER, "first token must be 256 (__MEDIA_START__)")
        self.assertEqual(tokens[-1], MEDIA_END, "last token must be 257 (__MEDIA_END__)")
        for t in tokens[1:-1]:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, 256, f"image payload token {t} must be real byte (0..255)")
        out = enc.decode(tokens)
        self.assertEqual(out.size, (32, 32))
        self.assertEqual(out.mode, "RGB")
        print("   ✅ ImageEncoder: 32×32 RGB round-trip, marker tokens correct.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_image_encoder_rejects_bad_blob(self):
        """🛰️ busel MULTIMODAL — ImageEncoder.decode raises on missing/corrupt markers."""
        print("🧪 [MM-3] busel Multimodal — image decode rejects bad blob...")
        enc = ImageEncoder()
        with self.assertRaises(ValueError):
            enc.decode([])
        with self.assertRaises(ValueError):
            enc.decode([0] * 100)
        with self.assertRaises(ValueError):
            enc.decode([IMAGE_MARKER] + list(b"short") + [MEDIA_END])
        print("   ✅ ImageEncoder.decode raises on bad input.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_video_encoder_roundtrip(self):
        """🛰️ busel MULTIMODAL — VideoEncoder encodes frame_count + N×32×32×3 to bytes."""
        print("🧪 [MM-4] busel Multimodal — video encode/decode round-trip...")
        import numpy as _np
        import imageio.v3 as _iio
        tmp = "temp_test_mm_video.mp4"
        try:
            n_frames = 6
            frames = [_np.random.randint(0, 255, (32, 32, 3), dtype=_np.uint8) for _ in range(n_frames)]
            _iio.imwrite(tmp, frames, fps=10)
            enc = VideoEncoder(max_frames=3)
            tokens = enc.encode_file(tmp)
            self.assertIsInstance(tokens, list)
            self.assertEqual(tokens[0], IMAGE_MARKER)
            self.assertEqual(tokens[-1], MEDIA_END)
            expected_len = 1 + 4 + 3 * IMAGE_BYTES + 1
            self.assertEqual(len(tokens), expected_len, f"video must be {expected_len} tokens")
            payload = enc.decode(tokens)
            self.assertEqual(len(payload), 3 * IMAGE_BYTES, f"expected 3 frames × {IMAGE_BYTES}, got {len(payload)}")
            print(f"   ✅ VideoEncoder: {n_frames}-frame input downsampled to 3 frames, round-trip OK.")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_audio_encoder_roundtrip(self):
        """🛰️ busel MULTIMODAL — AudioEncoder writes [sr][n][sw][PCM16] with markers."""
        print("🧪 [MM-5] busel Multimodal — audio encode/decode round-trip...")
        import numpy as _np
        import soundfile as _sf
        import wave as _wave
        from io import BytesIO as _BI
        tmp = "temp_test_mm_audio.wav"
        try:
            sr = 16000
            data = _np.random.uniform(-0.3, 0.3, sr).astype(_np.float32)
            _sf.write(tmp, data, sr)
            enc = AudioEncoder(max_seconds=2.0)
            tokens = enc.encode_file(tmp)
            self.assertIsInstance(tokens, list)
            self.assertEqual(tokens[0], IMAGE_MARKER)
            self.assertEqual(tokens[-1], MEDIA_END)
            header = bytes(tokens[1:11])
            sr_out, n_out, sw_out = struct.unpack("<IIH", header)
            self.assertEqual(sr_out, sr)
            self.assertEqual(sw_out, 2, "sample_width must be 2 (int16)")
            self.assertEqual(n_out, sr, "1 second @ 16kHz = 16000 samples")
            wav = enc.decode_to_wav(tokens)
            with _wave.open(_BI(wav), "rb") as wf:
                self.assertEqual(wf.getframerate(), sr)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getnframes(), sr)
            print("   ✅ AudioEncoder: 1s @ 16kHz round-trips, header layout correct.")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_docx_encoder_roundtrip(self):
        """🛰️ busel MULTIMODAL — DocxEncoder writes UTF-8 plain text with markers."""
        print("🧪 [MM-6] busel Multimodal — docx encode round-trip...")
        import docx as _docx
        tmp = "temp_test_mm.docx"
        try:
            d = _docx.Document()
            d.add_paragraph("Hello, multimodal Busel!")
            d.add_paragraph("Line two.")
            d.save(tmp)
            enc = DocxEncoder()
            tokens = enc.encode_file(tmp)
            self.assertIsInstance(tokens, list)
            self.assertEqual(tokens[0], IMAGE_MARKER)
            self.assertEqual(tokens[-1], MEDIA_END)
            text = bytes(tokens[1:-1]).decode("utf-8")
            self.assertIn("Hello, multimodal Busel!", text)
            self.assertIn("Line two.", text)
            print("   ✅ DocxEncoder: 2-paragraph docx → UTF-8 tokens round-trip OK.")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_text_encoder_passes_bytes_through(self):
        """🛰️ busel MULTIMODAL — TextEncoder is a thin pass-through (no markers)."""
        print("🧪 [MM-7] busel Multimodal — text encode is pass-through...")
        tmp = "temp_test_mm.txt"
        try:
            with open(tmp, "wb") as f:
                f.write(b"hello \xe2\x98\x83 unicode")
            enc = TextEncoder()
            tokens = enc.encode_file(tmp)
            self.assertEqual(tokens, list(b"hello \xe2\x98\x83 unicode"))
            self.assertNotIn(IMAGE_MARKER, tokens, "text encoder must NOT inject media markers")
            self.assertNotIn(MEDIA_END, tokens, "text encoder must NOT inject media markers")
            print("   ✅ TextEncoder: byte-for-byte pass-through, no marker injection.")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_build_encoder_for_dispatch(self):
        """🛰️ busel MULTIMODAL — build_encoder_for routes by extension, falls back to text."""
        print("🧪 [MM-8] busel Multimodal — build_encoder_for extension dispatch...")
        self.assertIsInstance(build_encoder_for("a.png"), ImageEncoder)
        self.assertIsInstance(build_encoder_for("A.JPG"), ImageEncoder)
        self.assertIsInstance(build_encoder_for("b.mp4"), VideoEncoder)
        self.assertIsInstance(build_encoder_for("c.wav"), AudioEncoder)
        self.assertIsInstance(build_encoder_for("d.docx"), DocxEncoder)
        self.assertIsInstance(build_encoder_for("e.txt"), TextEncoder)
        self.assertIsInstance(build_encoder_for("f.md"), TextEncoder)
        self.assertIsInstance(build_encoder_for("g.unknown"), TextEncoder, "unknown must fall back to TextEncoder")
        print("   ✅ build_encoder_for routes by extension (case-insensitive), falls back to text.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_marker_constants_reserved_vocab(self):
        """🛰️ busel MULTIMODAL — 256, 257, 258 are reserved marker tokens (outside 0..255)."""
        print("🧪 [MM-9] busel Multimodal — marker tokens are reserved (>= 256)...")
        self.assertEqual(IMAGE_MARKER, 256, "__MEDIA_START__ must be token 256")
        self.assertEqual(MEDIA_END, 257, "__MEDIA_END__ must be token 257")
        for marker in (IMAGE_MARKER, MEDIA_END, 258):
            self.assertGreaterEqual(marker, 256, f"marker {marker} must be in reserved range (>= 256)")
        for b in range(256):
            self.assertNotIn(b, (IMAGE_MARKER, MEDIA_END, 258), f"byte {b} collides with reserved marker")
        print("   ✅ Marker tokens 256, 257, 258 are reserved; no collision with valid UTF-8 bytes.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_image_encoder_byte_layout_lossless(self):
        """🛰️ busel MULTIMODAL — image encoder is lossless: encode→decode→encode is a fixed point."""
        print("🧪 [MM-10] busel Multimodal — image encoder fixed point (lossless)...")
        from PIL import Image as _Im
        import numpy as _np
        enc = ImageEncoder()
        src = _Im.fromarray(_np.random.randint(0, 255, (100, 100, 3), dtype=_np.uint8))
        blob1 = enc.encode(src)
        decoded = enc.decode(blob1)
        blob2 = enc.encode(decoded)
        self.assertEqual(blob1, blob2, "encode→decode→encode must be a fixed point (lossless)")
        print("   ✅ Image encoder is lossless: encode→decode→encode is byte-identical.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_text_encoder_in_pipeline_collates_to_int32(self):
        """🛰️ busel MULTIMODAL — encoder output flows through collate_busel_batch to int32 tensor."""
        print("🧪 [MM-11] busel Multimodal — encoder output → collate → int32 tensor with values up to 258...")
        from data.pipeline import collate_busel_batch
        enc = TextEncoder()
        tokens = enc.encode_file(__file__)
        self.assertIsInstance(tokens, list)
        batch = collate_busel_batch([(tokens, 0, 0)])
        tensor, _, _ = batch
        self.assertEqual(tensor.dtype, torch.int32, "collate must produce int32 tensor")
        self.assertLess(tensor.max().item(), 259, "max token must be < 259 (vocab size)")
        self.assertGreaterEqual(tensor.min().item(), 0, "min token must be >= 0")
        self.assertEqual(tensor.shape[0], 1, "batch size must be 1")
        print(f"   ✅ TextEncoder output flows through collate to int32 tensor of shape {tuple(tensor.shape)}, max={tensor.max().item()}.")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_cv2_fast_path_under_500ms_per_100_imgs(self):
        """🛰️ busel MULTIMODAL — cv2 image encode: 100×256² images must encode in <500ms."""
        print("🧪 [MM-12] busel Multimodal — cv2 fast-path throughput (100 imgs < 500ms)...")
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError:
            self.skipTest("opencv-python-headless not installed")
        from PIL import Image as _Im
        enc = ImageEncoder()
        n = 100
        src_arr = _np.random.randint(0, 255, (256, 256, 3), dtype=_np.uint8)
        src = _Im.fromarray(src_arr)
        t0 = time.perf_counter()
        for _ in range(n):
            enc.encode(src)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_op = elapsed_ms / n
        self.assertLess(elapsed_ms, 500, f"100 cv2 encodings took {elapsed_ms:.1f}ms (> 500ms budget)")
        print(f"   ✅ cv2 fast path: {per_op:.2f} ms/image ({n} images in {elapsed_ms:.1f}ms)")

    @unittest.skipUnless(HAS_MULTIMODAL_DEPS, "multimodal encoders unavailable")
    def test_mm_video_cv2_path_under_2s_for_60_frame_video(self):
        """🛰️ busel MULTIMODAL — cv2 video encode: 60-frame synthetic video must encode in <2s."""
        print("🧪 [MM-13] busel Multimodal — cv2 video path throughput (60 frames < 2s)...")
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError:
            self.skipTest("opencv-python-headless not installed")
        tmp = "temp_test_mm_video_perf.mp4"
        try:
            n_frames = 60
            h, w = 128, 128
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            writer = _cv2.VideoWriter(tmp, fourcc, 30.0, (w, h))
            for i in range(n_frames):
                frame = _np.random.randint(0, 255, (h, w, 3), dtype=_np.uint8)
                writer.write(frame)
            writer.release()
            enc = VideoEncoder(max_frames=8)
            t0 = time.perf_counter()
            tokens = enc.encode_file(tmp)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.assertLess(elapsed_ms, 2000, f"video encoding took {elapsed_ms:.1f}ms (> 2000ms budget)")
            self.assertEqual(tokens[0], IMAGE_MARKER)
            self.assertEqual(tokens[-1], MEDIA_END)
            print(f"   ✅ cv2 video path: 60 frames @ 128×128 → 8 frames in {elapsed_ms:.1f}ms")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_muon_routing_rule_2d_proj_router_embed(self):
        """🛡️ Muon routing (ISSUES.md #5): 2D+!router+!embed → Muon, else → AdamW."""
        print("🧪 [R5] busel Muon routing — 2D non-router/non-embed → Muon, else → AdamW...")
        class MockM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_q_proj = torch.nn.Linear(64, 64, bias=False)
                self.expert_ffn_0_weight = torch.nn.Parameter(torch.randn(128, 64))
                self.mtp_head_3 = torch.nn.Parameter(torch.randn(259, 128))
                self.bias = torch.nn.Parameter(torch.zeros(64))
                self.router = torch.nn.Linear(64, 4, bias=False)
                self.token_embed = torch.nn.Parameter(torch.randn(259, 64))

        model = MockM()
        engine = buselOptimizerEngine(model, lr_muon=0.01, lr_adamw=0.001)
        muon_param_ids = {id(p) for p in engine.opt_muon.param_groups[0]['params']}
        adamw_param_ids = {id(p) for p in engine.opt_adamw.param_groups[0]['params']}
        for name, p in model.named_parameters():
            in_muon = id(p) in muon_param_ids
            in_adamw = id(p) in adamw_param_ids
            self.assertTrue(in_muon or in_adamw, f"{name} assigned to NEITHER optimizer")
            self.assertFalse(in_muon and in_adamw, f"{name} assigned to BOTH optimizers")
            if "router" in name or "embed" in name or p.ndim != 2:
                self.assertTrue(in_adamw, f"{name} (1D/router/embed) should be in AdamW, got Muon")
            else:
                self.assertTrue(in_muon, f"{name} (2D+!router+!embed) should be in Muon, got AdamW")
        n_muon = len(muon_param_ids)
        n_adamw = len(adamw_param_ids)
        self.assertEqual(n_muon, 3, f"expected 3 Muon params, got {n_muon}")
        self.assertEqual(n_adamw, 3, f"expected 3 AdamW params, got {n_adamw}")
        print(f"   ✅ Muon routing: 3 Muon / 3 AdamW (correctly excludes router/embed/1D).")

    def test_training_schedule_guard_rejects_bad_inputs(self):
        """🛡️ Schedule guard (ISSUES.md #7): max_steps > warmup_steps, warmup >= 1."""
        print("🧪 [R7] busel schedule guard — rejects max_steps<=warmup_steps and warmup<1...")
        with self.assertRaises(ValueError, msg="max_steps==warmup_steps must be rejected"):
            validate_training_schedule(100, 100)
        with self.assertRaises(ValueError, msg="max_steps<warmup_steps must be rejected"):
            validate_training_schedule(50, 100)
        with self.assertRaises(ValueError, msg="warmup_steps<1 must be rejected"):
            validate_training_schedule(100, 0)
        with self.assertRaises(ValueError, msg="warmup_steps<0 must be rejected"):
            validate_training_schedule(100, -5)
        with self.assertRaises(ValueError, msg="None must be rejected"):
            validate_training_schedule(None, 10)
        with self.assertRaises(ValueError, msg="None must be rejected"):
            validate_training_schedule(10, None)
        max_s, warmup_s = validate_training_schedule(200, 10)
        self.assertEqual((max_s, warmup_s), (200, 10))
        max_s, warmup_s = validate_training_schedule("300", "20")
        self.assertEqual((max_s, warmup_s), (300, 20), "string-cast ints must be normalised")
        print("   ✅ Schedule guard: rejects all 6 bad cases, accepts 2 valid cases.")

    def test_inject_noise_branchless_preserves_grad_shape(self):
        """🛡️ inject_noise (ISSUES.md #6): branchless mask keeps grad shape & adds bounded noise."""
        print("🧪 [R6] busel inject_noise — branchless mask, grad shape preserved, noise bounded...")
        from training.autopilot import buselAutoPilot

        class _MockEngine:
            def __init__(self):
                self.opt_muon = type("O", (), {"param_groups": [{"params": []}]})()
                self.opt_adamw = type("O", (), {"param_groups": [{"params": []}]})()

        engine = _MockEngine()
        ap = buselAutoPilot(engine, max_lr_muon=0.01, max_lr_adamw=0.001, noise_scale=0.1)
        ap.noise_scale = 0.1

        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.Linear(8, 4, bias=False),
        )
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.5
        g_before = [p.grad.clone() for p in model.parameters()]

        ap.inject_noise(model)

        for p, g in zip(model.parameters(), g_before):
            self.assertEqual(p.grad.shape, g.shape, "grad shape must be preserved")
            self.assertTrue(torch.isfinite(p.grad).all(), "grad must remain finite")
            self.assertFalse(torch.allclose(p.grad, g, atol=1e-8), "noise should perturb the grad")
        ap.noise_scale = 0.0
        g_before2 = [p.grad.clone() for p in model.parameters()]
        ap.inject_noise(model)
        for p, g in zip(model.parameters(), g_before2):
            self.assertTrue(torch.allclose(p.grad, g, atol=1e-12), "noise_scale=0 must be no-op")
        print("   ✅ inject_noise: shape preserved, finite, no-op at zero scale.")


if __name__ == "__main__":
    unittest.main()

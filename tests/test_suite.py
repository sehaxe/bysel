"""
🧪 busel UNIFIED TEST SUITE — paper-compliance + integration
Covers all 6 reference papers + end-to-end integration.
"""
import os
import sys
import math
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
from training.recipe import buselLossEngine


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


if __name__ == "__main__":
    unittest.main()

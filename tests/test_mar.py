"""🧪 busel mAR unit tests — Sinkhorn-Knopp + Manifold-Constrained Attention Residuals.

Validates:
  • sinkhorn_knopp produces doubly-stochastic matrices (rows AND cols sum to 1)
  • Identity-initialized H matrix → H ≈ I (mHC's identity-mapping property)
  • Output shape is preserved through mAR
  • Backward pass works (gradients flow)
  • buselModel accepts n_hyper config and produces same-shape output
"""
import math
import sys
import unittest

import torch

from model.backbone import ManifoldConstrainedAttnRes, buselModel


class _MockConfig:
    def __init__(self, **kw):
        self.d_model = kw.get("d_model", 128)
        self.n_layers = kw.get("n_layers", 3)
        self.n_heads = kw.get("n_heads", 4)
        self.expert_hidden = kw.get("expert_hidden", 256)
        self.num_experts = kw.get("num_experts", 2)
        self.top_k = kw.get("top_k", 2)
        self.vocab_size = kw.get("vocab_size", 259)
        self.n_hyper = kw.get("n_hyper", 2)


class TestSinkhornKnopp(unittest.TestCase):
    """sinkhorn_knopp must project M onto the Birkhoff polytope."""

    def test_rows_sum_to_one(self):
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(2, 4, 3, 3)
        H = mar.sinkhorn_knopp(M)
        row_sums = H.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)

    def test_cols_sum_to_one(self):
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(2, 4, 3, 3)
        H = mar.sinkhorn_knopp(M)
        col_sums = H.sum(dim=-2)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)

    def test_all_non_negative(self):
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        M = torch.randn(8, 16, 2, 2) * 100.0
        H = mar.sinkhorn_knopp(M)
        self.assertTrue((H >= 0).all())

    def test_extreme_inputs_no_overflow(self):
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        M = torch.randn(1, 1, 2, 2) * 1000.0
        H = mar.sinkhorn_knopp(M)
        self.assertFalse(torch.isnan(H).any())
        self.assertFalse(torch.isinf(H).any())

    def test_more_iters_tighter_to_birkhoff(self):
        mar = ManifoldConstrainedAttnRes(d_model=129, n_hyper=3, n_sinkhorn_iters=20)
        M = torch.randn(4, 5, 3, 3)
        H = mar.sinkhorn_knopp(M, n_iters=20)
        row_dev = (H.sum(dim=-1) - 1.0).abs().max()
        col_dev = (H.sum(dim=-2) - 1.0).abs().max()
        self.assertLess(row_dev.item(), 1e-3)
        self.assertLess(col_dev.item(), 1e-3)


class TestIdentityInit(unittest.TestCase):
    """At init, with no learned q/k influence, H should ≈ I (mHC's identity-mapping)."""

    def test_h_is_near_identity(self):
        d_model, n_hyper = 128, 2
        mar = ManifoldConstrainedAttnRes(d_model=d_model, n_hyper=n_hyper, n_sinkhorn_iters=5)
        mar.eval()
        torch.manual_seed(0)
        x = torch.zeros(1, 4, d_model)
        streams = tuple(torch.zeros(1, 4, d_model) for _ in range(n_hyper))
        with torch.no_grad():
            y = mar(x, streams)
        with torch.no_grad():
            q = mar.q_proj(x).view(1, 4, n_hyper, mar.d_head)
            ks = [mar.k_proj(s) for s in streams]
            k_stack = torch.stack(ks, dim=2)
            H_logits = torch.einsum('btqd,btkd->btqk', q, k_stack) / math.sqrt(mar.d_head)
            H_logits = H_logits + mar.identity_bias
            H = mar.sinkhorn_knopp(H_logits)
        diag = H.diagonal(dim1=-2, dim2=-1)
        diag_matrix = torch.diag_embed(diag)
        off_diag = H - diag_matrix
        self.assertGreater(diag.mean().item(), off_diag.abs().mean().item())


class TestForwardShape(unittest.TestCase):
    """mAR must preserve [B, T, d_model] shape through the forward pass."""

    def test_output_shape(self):
        d_model, n_hyper, B, T = 128, 2, 4, 16
        mar = ManifoldConstrainedAttnRes(d_model=d_model, n_hyper=n_hyper)
        x = torch.randn(B, T, d_model)
        streams = tuple(torch.randn(B, T, d_model) for _ in range(n_hyper))
        y = mar(x, streams)
        self.assertEqual(y.shape, (B, T, d_model))

    def test_streams_mismatch_raises(self):
        mar = ManifoldConstrainedAttnRes(d_model=128, n_hyper=2)
        x = torch.randn(2, 4, 128)
        streams = (torch.randn(2, 4, 128),)
        with self.assertRaises(ValueError):
            mar(x, streams)

    def test_d_model_not_divisible_by_n_hyper_raises(self):
        with self.assertRaises(ValueError):
            ManifoldConstrainedAttnRes(d_model=127, n_hyper=2)


class TestBackward(unittest.TestCase):
    """Gradients must flow through mAR (Sinkhorn-Knopp is differentiable via exp)."""

    def test_gradients_flow(self):
        d_model, n_hyper, B, T = 64, 2, 2, 8
        mar = ManifoldConstrainedAttnRes(d_model=d_model, n_hyper=n_hyper)
        x = torch.randn(B, T, d_model, requires_grad=True)
        streams = tuple(torch.randn(B, T, d_model, requires_grad=True) for _ in range(n_hyper))
        y = mar(x, streams)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        for s in streams:
            self.assertIsNotNone(s.grad)
            self.assertFalse(torch.isnan(s.grad).any())

    def test_temperature_gets_grad(self):
        mar = ManifoldConstrainedAttnRes(d_model=64, n_hyper=2)
        x = torch.randn(2, 4, 64)
        streams = tuple(torch.randn(2, 4, 64) for _ in range(2))
        y = mar(x, streams)
        y.sum().backward()
        self.assertIsNotNone(mar.temperature.grad)
        self.assertFalse(torch.isnan(mar.temperature.grad).any())


class TestBuselModelIntegration(unittest.TestCase):
    """buselModel with n_hyper streams must produce correct output shapes."""

    def test_n_hyper_2(self):
        cfg = _MockConfig(d_model=128, n_layers=3, n_heads=4, n_hyper=2)
        model = buselModel(cfg)
        B, T = 2, 8
        hidden = torch.randn(B, T, cfg.d_model)
        mtp, aux = model(hidden)
        self.assertEqual(mtp[0].shape, (B, T, 259))
        self.assertEqual(aux.shape, ())

    def test_n_hyper_4(self):
        cfg = _MockConfig(d_model=128, n_layers=3, n_heads=4, n_hyper=4)
        model = buselModel(cfg)
        B, T = 2, 8
        hidden = torch.randn(B, T, cfg.d_model)
        mtp, aux = model(hidden)
        self.assertEqual(mtp[0].shape, (B, T, 259))


if __name__ == "__main__":
    unittest.main()

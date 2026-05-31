"""
🧪 BYSEL UNIFIED TEST SUITE v4.6 (FINAL)
Унифицированный тест-сьют с нативной поддержкой SwishGLUClamped из BitNet v2.
"""

import os
import sys
import unittest
import json
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bysel_rust_io
from data.pipeline import get_bysel_dataloader, collate_bysel_batch
from model.patching import StridedFastBLTPatcher
from model.layers import BitLinear_a4_8, RMSNorm, SwishGLUClamped
from model.attention import stable_gdn2_recurrent_jit, BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanExpertFFN, BulbaTernaryTitanMoE
from model.backbone import ByselModel
from training.optimizer import _compiled_newton_schulz, ByselOptimizerEngine
from training.recipe import ByselLossEngine


class TestByselFramework(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🚀 Running Bysel Test Suite on device: {cls.device.upper()}\n" + "="*80)

    def test_rust_io_streamer(self):
        print("🧪 [TEST 1/8] Testing Rust ByteStreamer...")
        temp_file = "temp_test_rust_io.txt"
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        test_data = "Hello from Bysel Rust IO! " * 350
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(test_data)
            
        try:
            streamer = bysel_rust_io.ByteStreamer(temp_file, 8, 0)
            chunk1 = streamer.next_chunk()
            self.assertEqual(len(chunk1), 8)
            self.assertEqual(bytes(chunk1).decode('utf-8', errors='ignore'), "Hello fr")
            
            chunk2 = streamer.next_chunk()
            self.assertEqual(len(chunk2), 8)
            self.assertEqual(bytes(chunk2).decode('utf-8', errors='ignore'), "om Bysel")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("   ✅ Rust ByteStreamer passed.")

    def test_rust_binary_packer(self):
        print("🧪 [TEST 2/8] Testing Rust Binary Packer...")
        temp_bin = "temp_test_packer.bin"
        
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
            
        test_bytes = [10, 20, 30, 40, 255, 0]
        
        try:
            bysel_rust_io.append_to_binary_file(temp_bin, test_bytes)
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
        print("🧪 [TEST 3/8] Testing Rust Ternary CPU Inference...")
        x = [1.0, -1.0, 2.0]
        w = [
            1,  0, -1, 
            0,  1,  1  
        ]
        
        expected_y = [-1.0, 1.0]
        actual_y = bysel_rust_io.ternary_matmul_cpu(x, w, 2, 3)
        
        self.assertEqual(actual_y, expected_y)
        print("   ✅ Rust Ternary CPU Inference passed.")

    def test_bitlinear_quantization(self):
        print("🧪 [TEST 4/8] Testing BitLinear 1.58b Quantization...")
        linear = BitLinear_a4_8(64, 128).to(self.device)
        x = torch.randn(2, 64, device=self.device)
        
        out = linear(x)
        self.assertEqual(out.shape, (2, 128))
        self.assertFalse(torch.isnan(out).any())
        print("   ✅ BitLinear Quantization passed.")

    def test_jit_gdn2_attention(self):
        print("🧪 [TEST 5/8] Testing Stable JIT GDN-2 Loop...")
        q = torch.randn(2, 128, 4, 64, device=self.device)
        k = torch.randn(2, 128, 4, 64, device=self.device)
        v = torch.randn(2, 128, 4, 64, device=self.device)
        b = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        w = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        alpha = torch.sigmoid(torch.randn(2, 128, 4, 64, device=self.device))
        
        # L2-нормализация ключей и запросов по размерности каналов
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        
        with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device == "mps" else torch.bfloat16):
            out = stable_gdn2_recurrent_jit(q, k, v, b, w, alpha)
            
        self.assertEqual(out.shape, (2, 128, 256))
        self.assertFalse(torch.isnan(out).any())
        print("   ✅ Stable JIT GDN-2 Loop passed.")

    def test_fused_glu_and_expert_ffn(self):
        print("🧪 [TEST 6/8] Testing Fused Gate-Up Projections...")
        x = torch.randn(2, 32, 256, device=self.device)
        
        # ReLU2GLUClamped заменен на SwishGLUClamped по спецификации BitNet v2
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
        print("🧪 [TEST 7/8] Testing Muon Transpose Trick on Tall Matrices...")
        X = torch.randn(512, 256, device=self.device)
        O_t = _compiled_newton_schulz(X, steps=5)
        
        self.assertEqual(O_t.shape, (512, 256))
        self.assertFalse(torch.isnan(O_t).any(), "O_t contains NaNs!")
        print("   ✅ Muon Transpose Trick passed.")

    def test_complete_backbone_and_gradients(self):
        print("🧪 [TEST 8/8] Testing Complete Backbone & Backpropagation...")
        
        class MockConfig:
            vocab_size = 259
            d_model = 256
            n_layers = 4  
            n_heads = 4
            expert_hidden = 512
            num_experts = 4
            top_k = 2
            
        cfg = MockConfig()
        patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(self.device)
        model = ByselModel(cfg).to(self.device)
        
        opt_engine = ByselOptimizerEngine(model, lr_muon=0.0004, lr_adamw=0.00004)
        loss_engine = ByselLossEngine(cfg.vocab_size)
        
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


if __name__ == "__main__":
    unittest.main()
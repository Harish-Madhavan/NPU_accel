import torch
import unittest
import intel_npu_acceleration
import time
import torch.nn as nn
import torch.nn.functional as F

# Custom _torch_rmsnorm for reference, matching the one in __init__.py
def _torch_rmsnorm(input, weight, eps):
    # This is a simplified RMSNorm, might need adjustment for full compatibility
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return input * weight


class TestIntelNPULibLLM(unittest.TestCase):
    def setUp(self):
        self.avail = intel_npu_acceleration.is_available()
        if not self.avail:
            print("WARNING: NPU not available.")

    def test_activation_ops(self):
        x = torch.randn(10, 10)
        
        # ReLU
        res = intel_npu_acceleration.relu(x)
        self.assertTrue(torch.allclose(res, torch.relu(x), rtol=1e-2, atol=1e-2))
        
        # GELU
        res = intel_npu_acceleration.gelu(x)
        self.assertTrue(torch.allclose(res, F.gelu(x), rtol=1e-2, atol=1e-2))
        
        # Softmax
        res = intel_npu_acceleration.softmax(x, dim=-1)
        self.assertTrue(torch.allclose(res, F.softmax(x, dim=-1), rtol=1e-2, atol=1e-2))

        # SiLU
        res = intel_npu_acceleration.silu(x)
        self.assertTrue(torch.allclose(res, F.silu(x), rtol=1e-2, atol=1e-2))

    def test_silu_op(self):
        x = torch.randn(5, 7, 128)
        res = intel_npu_acceleration.silu(x)
        expected = F.silu(x)
        self.assertTrue(torch.allclose(res, expected, rtol=1e-2, atol=1e-2))

    def test_rmsnorm_op(self):
        # Test Case 1: Basic
        input1 = torch.randn(2, 4, 8)
        weight1 = torch.ones(8)
        eps1 = 1e-6
        npu_res1 = intel_npu_acceleration.rmsnorm(input1, weight1, eps1)
        cpu_res1 = _torch_rmsnorm(input1, weight1, eps1)
        self.assertTrue(torch.allclose(npu_res1, cpu_res1, rtol=1e-2, atol=1e-2))

        # Test Case 2: Different weight
        input2 = torch.randn(1, 10, 256)
        weight2 = torch.randn(256)
        eps2 = 1e-5
        npu_res2 = intel_npu_acceleration.rmsnorm(input2, weight2, eps2)
        cpu_res2 = _torch_rmsnorm(input2, weight2, eps2)
        self.assertTrue(torch.allclose(npu_res2, cpu_res2, rtol=1e-2, atol=1e-2))

        # Test Case 3: Larger epsilon
        input3 = torch.randn(3, 5, 64)
        weight3 = torch.rand(64) + 0.1 # Ensure weight is not too small
        eps3 = 1e-3
        npu_res3 = intel_npu_acceleration.rmsnorm(input3, weight3, eps3)
        cpu_res3 = _torch_rmsnorm(input3, weight3, eps3)
        self.assertTrue(torch.allclose(npu_res3, cpu_res3, rtol=1e-2, atol=1e-2))

    def test_transformer_block(self):
        class TinyTransformerBlock(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_model * 4)
                self.activation = nn.GELU()
                self.linear2 = nn.Linear(d_model * 4, d_model)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x

        d_model = 16
        model = TinyTransformerBlock(d_model)
        model.eval()
        
        compiled_model = intel_npu_acceleration.compile(model)
        
        x = torch.randn(1, 8, d_model) # Batch 1, Seq 8, Dim 16
        
        # Warmup
        _ = compiled_model(x)
        
        start = time.time()
        out_npu = compiled_model(x)
        print(f"Transformer Block Time: {time.time() - start:.4f}s")
        
        out_cpu = model(x)
        
        # The composite error might be higher, checking
        diff = (out_npu - out_cpu).abs().max().item()
        print(f"Transformer Block Max Diff: {diff}")
        self.assertTrue(torch.allclose(out_npu, out_cpu, rtol=1e-1, atol=1e-1))

if __name__ == '__main__':
    unittest.main()

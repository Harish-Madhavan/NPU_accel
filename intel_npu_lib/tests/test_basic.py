import torch
import unittest
import intel_npu_acceleration
import time

class TestIntelNPULib(unittest.TestCase):
    def setUp(self):
        self.avail = intel_npu_acceleration.is_available()
        if not self.avail:
            print("WARNING: NPU not available, tests might fallback or fail if strict.")

    def test_basic_ops(self):
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        # Add
        res_add = intel_npu_acceleration.add(a, b)
        self.assertTrue(torch.allclose(res_add, a + b, rtol=1e-2, atol=1e-2))

        # Sub
        res_sub = intel_npu_acceleration.sub(a, b)
        self.assertTrue(torch.allclose(res_sub, a - b, rtol=1e-2, atol=1e-2))
        
        # Mul
        res_mul = intel_npu_acceleration.mul(a, b)
        self.assertTrue(torch.allclose(res_mul, a * b, rtol=1e-2, atol=1e-2))
        
        # Div
        # Avoid division by zero
        b_safe = b + 0.1 
        res_div = intel_npu_acceleration.div(a, b_safe)
        # Relaxed tolerance for NPU division
        self.assertTrue(torch.allclose(res_div, a / b_safe, rtol=1e-2, atol=1e-2))

    def test_matmul(self):
        a = torch.randn(4, 5)
        b = torch.randn(5, 3)
        res = intel_npu_acceleration.matmul(a, b)
        expected = torch.matmul(a, b)
        self.assertTrue(torch.allclose(res, expected, rtol=1e-2, atol=1e-2))

    def test_integration_compile(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x, y):
                z = torch.add(x, y)
                w = torch.mul(z, x)
                return w

        model = SimpleModel()
        compiled_model = intel_npu_acceleration.compile(model)
        
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        
        # Warmup (compilation happens here)
        start = time.time()
        out_compiled = compiled_model(x, y)
        end = time.time()
        print(f"First run (compile + exec): {end - start:.4f}s")
        
        # Second run (should be faster due to caching in ops.cpp)
        start = time.time()
        out_compiled_2 = compiled_model(x, y)
        end = time.time()
        print(f"Second run (cached exec): {end - start:.4f}s")

        out_original = model(x, y)
        
        self.assertTrue(torch.allclose(out_compiled, out_original, rtol=1e-2, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
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

    def test_activations_compile(self):
        class ActivationsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()
                self.relu6 = torch.nn.ReLU6()
                self.leaky_relu = torch.nn.LeakyReLU(0.05)
                self.hardtanh = torch.nn.Hardtanh(-2.0, 2.0)

            def forward(self, x):
                # Test functional, method, and module versions of Sigmoid & Tanh
                s1 = self.sigmoid(x)
                s2 = torch.sigmoid(s1)
                s3 = s2.sigmoid()

                t1 = self.tanh(s3)
                t2 = torch.tanh(t1)
                t3 = t2.tanh()

                # Test functional and module versions of ReLU6 & LeakyReLU
                r1 = self.relu6(t3)
                r2 = torch.nn.functional.relu6(r1)

                l1 = self.leaky_relu(r2)
                l2 = torch.nn.functional.leaky_relu(l1, negative_slope=0.05)

                # Test clamp (functional & method) and Hardtanh module
                c1 = self.hardtanh(l2)
                c2 = torch.clamp(c1, min=-1.5, max=1.5)
                c3 = torch.nn.functional.hardtanh(c2, min_val=-1.2, max_val=1.2)
                c4 = c3.clamp(min=-1.0, max=1.0)
                return c4

        model = ActivationsModel()
        x = torch.randn(2, 4)
        compiled_model = intel_npu_acceleration.compile(model, x, strict=True)

        out_compiled = compiled_model(x)
        out_expected = model(x)
        self.assertTrue(
            torch.allclose(out_compiled, out_expected, rtol=1e-2, atol=1e-2)
        )

    def test_integration_compile(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x, y):
                z = torch.add(x, y)
                w = torch.mul(z, x)
                return w

        model = SimpleModel()

        x = torch.randn(10, 10)
        y = torch.randn(10, 10)

        # New API requires example_input
        compiled_model = intel_npu_acceleration.compile(model, (x, y))

        # Warmup (compilation happens inside compile_to_npu now, not lazy)
        start = time.time()
        out_compiled = compiled_model(x, y)
        end = time.time()
        print(f"First run (exec): {end - start:.4f}s")

        # Second run
        start = time.time()
        compiled_model(x, y)
        end = time.time()
        print(f"Second run (cached exec): {end - start:.4f}s")

        out_original = model(x, y)

        self.assertTrue(
            torch.allclose(out_compiled, out_original, rtol=1e-2, atol=1e-2)
        )

    def test_cache_management(self):
        # 1. Test get_cache_size returns expected tuple structure
        size, count = intel_npu_acceleration.get_cache_size()
        self.assertIsInstance(size, int)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(size, 0)
        self.assertGreaterEqual(count, 0)

        # 2. Test clean_old_cache runs without error
        intel_npu_acceleration.clean_old_cache(max_size_mb=10, max_files=5)

        # 3. Test clear_cache runs without error
        intel_npu_acceleration.clear_cache()
        
        # After clearing, verify no exceptions occurred and size is less than or equal to original.
        # (Some files may be locked by active OpenVINO execution in other parallel tests on Windows)
        new_size, new_count = intel_npu_acceleration.get_cache_size()
        self.assertLessEqual(new_size, size)
        self.assertLessEqual(new_count, count)


if __name__ == "__main__":
    unittest.main()

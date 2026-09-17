import time
import unittest

import torch
import torch.nn as nn

import intel_npu_acceleration as npu
from intel_npu_acceleration.frontend import NPUCompilationError


class TestFXTracing(unittest.TestCase):
    def test_fx_tracing_dtypes(self):
        # 1. Test float32 compilation
        class ModelF32(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        m_f32 = ModelF32()
        x_f32 = torch.randn(5, 5, dtype=torch.float32)
        y_f32 = torch.randn(5, 5, dtype=torch.float32)

        n_f32 = npu.compile(m_f32, (x_f32, y_f32))
        res_n = n_f32(x_f32, y_f32)
        self.assertTrue(torch.allclose(res_n, x_f32 + y_f32, atol=1e-2, rtol=1e-2))

        # 2. Test float16 compilation
        class ModelF16(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        m_f16 = ModelF16()
        x_f16 = torch.randn(5, 5, dtype=torch.float16)
        y_f16 = torch.randn(5, 5, dtype=torch.float16)

        n_f16 = npu.compile(m_f16, (x_f16, y_f16))
        res_n16 = n_f16(x_f16, y_f16)
        self.assertTrue(
            torch.allclose(
                res_n16.float(), (x_f16 + y_f16).float(), atol=1e-2, rtol=1e-2
            )
        )

    def test_fx_tracing_caching(self):
        class SimpleModel(nn.Module):
            def forward(self, x):
                return torch.mul(x, 2.0)

        m = SimpleModel()
        x = torch.randn(10, 10)

        # First compilation (Graph creation & OpenVINO device compiler overhead)
        t0 = time.time()
        _ = npu.compile(m, x)
        t_first = time.time() - t0

        # Second compilation (Must hit front-end Python dict cache instantly)
        t0 = time.time()
        _ = npu.compile(m, x)
        t_second = time.time() - t0

        print(
            f"\nFX Tracing Caching benchmark - First Compile: {t_first:.4f}s | Second (Cache): {t_second:.4f}s"
        )
        # Python cache lookup is extremely fast (< 1-2 ms), while actual compilation takes 10+ ms.
        self.assertTrue(t_second < 0.05 or t_second < t_first * 0.1)

    def test_fx_tracing_varying_shapes(self):
        class ShapeModel(nn.Module):
            def forward(self, x):
                return torch.reshape(x, (2, -1))

        m = ShapeModel()

        # Compile with shape (2, 4)
        x1 = torch.randn(2, 4)
        n1 = npu.compile(m, x1)
        res1 = n1(x1)
        self.assertEqual(res1.shape, (2, 4))

        # Compile with shape (4, 4)
        x2 = torch.randn(4, 4)
        n2 = npu.compile(m, x2)
        res2 = n2(x2)
        self.assertEqual(res2.shape, (2, 8))

    def test_fx_tracing_zero_dim_inputs(self):
        # Verify tracing with zero-dimensional scalar input tensors
        class ScalarModel(nn.Module):
            def forward(self, x, scalar):
                return x + scalar

        m = ScalarModel()
        x = torch.randn(5, 5)
        scalar = torch.tensor(3.0)  # 0D scalar

        try:
            n_model = npu.compile(m, (x, scalar))
            res = n_model(x, scalar)
            self.assertTrue(torch.allclose(res, x + 3.0, atol=1e-2, rtol=1e-2))
        except Exception as e:
            self.fail(f"Zero-dimensional scalar input compilation failed: {e}")

    def test_fx_tracing_invalid_ops(self):
        # Tracing an unsupported operator must raise NPUCompilationError
        class InvalidModel(nn.Module):
            def forward(self, x):
                # torch.fft.fft is an unsupported complex/frequency domain operator
                return torch.fft.fft(x)

        m = InvalidModel()
        x = torch.randn(8)

        with self.assertRaises(NPUCompilationError):
            npu.compile(m, x, strict=True)


if __name__ == "__main__":
    unittest.main()

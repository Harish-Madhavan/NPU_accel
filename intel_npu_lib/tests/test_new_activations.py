import unittest

import torch
import torch.nn as nn

from intel_npu_acceleration import compile


class StandardActModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.gelu_tanh = nn.GELU(approximate="tanh")
        self.silu = nn.SiLU()

    def forward(self, x):
        r = self.relu(x)
        g1 = self.gelu(r)
        g2 = self.gelu_tanh(g1)
        s = self.silu(g2)
        return s


class HardActivationsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid_mod = nn.Hardsigmoid()
        self.hardswish_mod = nn.Hardswish()

    def forward(self, x):
        # 1. Hardsigmoid functional and module
        hs1 = torch.nn.functional.hardsigmoid(x)
        hs2 = self.hardsigmoid_mod(hs1)

        # 2. Hardswish functional and module
        hw1 = torch.nn.functional.hardswish(hs2)
        hw2 = self.hardswish_mod(hw1)

        return hw2


class TestNewActivations(unittest.TestCase):
    def setUp(self):
        # Clean the compiler graph cache before each test
        import intel_npu_acceleration.frontend as frontend
        frontend._GRAPH_CACHE.clear()

    def test_standard_act_modules(self):
        model = StandardActModules()
        model.eval()

        x = torch.randn(2, 8, 16)

        try:
            npu_model = compile(model, x, strict=True)
        except Exception as e:
            self.fail(f"Compilation failed for standard module activations: {e}")

        out_cpu = model(x)
        out_npu = npu_model(x)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

    def test_hard_activations(self):
        model = HardActivationsModel()
        model.eval()

        x = torch.randn(2, 4, 8)

        try:
            npu_model = compile(model, x, strict=True)
        except Exception as e:
            self.fail(f"Compilation failed for Hardsigmoid and Hardswish: {e}")

        out_cpu = model(x)
        out_npu = npu_model(x)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()

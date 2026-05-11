import unittest
import torch
import torch.nn as nn
from intel_npu_acceleration import compile_to_npu


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TestMLP(unittest.TestCase):
    def test_mlp_inference(self):
        model = SimpleMLP()
        model.eval()

        x = torch.randn(1, 16)

        try:
            npu_model = compile_to_npu(model, x)
        except Exception as e:
            self.fail(f"Compilation failed: {e}")

        out_npu = npu_model(x)
        out_cpu = model(x)

        self.assertTrue(
            torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3),
            f"Output mismatch. Max diff: {(out_npu - out_cpu).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()

import pytest
import torch

from intel_npu_acceleration import compile_to_npu
from tests.helpers import make_simple_mlp


class TestMLP:
    def test_mlp_inference(self):
        model = make_simple_mlp()
        model.eval()

        x = torch.randn(1, 16)

        try:
            npu_model = compile_to_npu(model, x)
        except Exception as e:
            pytest.fail(f"Compilation failed: {e}")

        out_npu = npu_model(x)
        out_cpu = model(x)

        assert torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3), (
            f"Output mismatch. Max diff: {(out_npu - out_cpu).abs().max().item()}"
        )

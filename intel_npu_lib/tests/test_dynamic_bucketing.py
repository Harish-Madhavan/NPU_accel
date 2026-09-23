import pytest
import torch
import torch.nn as nn

from intel_npu_acceleration import compile


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return torch.relu(self.fc(x))


class HybridUnsupportedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        # erf is unsupported on NPU and triggers CPU fallback
        return torch.erf(torch.relu(self.fc(x)))


class TestDynamicBucketing:
    def test_sync_bucketing(self):
        model = SimpleMLP()
        model.eval()

        example_input = torch.randn(1, 8, 32)

        # Compile with dynamic shape bucketing enabled
        npu_model = compile(
            model,
            example_input,
            dynamic_buckets=True,
            bucket_sizes=[8, 16, 32, 64],
            dynamic_dim=1,
        )

        # Test various dynamic sequence lengths
        for seq_len in [5, 12, 37]:
            x = torch.randn(1, seq_len, 32)
            out_cpu = model(x)
            out_npu = npu_model(x)

            assert out_npu.shape == out_cpu.shape
            assert torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2)

    def test_async_bucketing(self):
        model = SimpleMLP()
        model.eval()

        example_input = torch.randn(1, 8, 32)

        npu_model = compile(
            model,
            example_input,
            dynamic_buckets=True,
            bucket_sizes=[8, 16, 32, 64],
            dynamic_dim=1,
            num_streams=2,
            performance_hint="THROUGHPUT",
        )

        # Async request 1: seq_len = 5
        x1 = torch.randn(1, 5, 32)
        h1 = npu_model.infer_async(x1)

        # Async request 2: seq_len = 12
        x2 = torch.randn(1, 12, 32)
        h2 = npu_model.infer_async(x2)

        # Wait and verify
        out_npu1 = npu_model.wait_async(h1)
        out_cpu1 = model(x1)
        assert out_npu1.shape == out_cpu1.shape
        assert torch.allclose(out_npu1, out_cpu1, atol=1e-2, rtol=1e-2)

        out_npu2 = npu_model.wait_async(h2)
        out_cpu2 = model(x2)
        assert out_npu2.shape == out_cpu2.shape
        assert torch.allclose(out_npu2, out_cpu2, atol=1e-2, rtol=1e-2)

    def test_zero_copy_option(self):
        model = SimpleMLP()
        model.eval()

        example_input = torch.randn(1, 8, 32)

        # Compile with clone_outputs=False (zero-copy mode)
        npu_model = compile(
            model,
            example_input,
            clone_outputs=False,
        )

        x = torch.randn(1, 8, 32)
        out_cpu = model(x)
        out_npu = npu_model(x)

        assert torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [7, 15])
    def test_hybrid_fallback_with_bucketing(self, seq_len):
        model = HybridUnsupportedMLP()
        model.eval()

        example_input = torch.randn(1, 8, 32)

        # Compile with dynamic shape bucketing + hybrid fallback enabled
        # (strict=False ensures CPU fallback is permitted)
        npu_model = compile(
            model,
            example_input,
            strict=False,
            dynamic_buckets=True,
            bucket_sizes=[8, 16, 32, 64],
            dynamic_dim=1,
        )

        x = torch.randn(1, seq_len, 32)
        out_cpu = model(x)
        out_npu = npu_model(x)

        assert out_npu.shape == out_cpu.shape
        assert torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2)

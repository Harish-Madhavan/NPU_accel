import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import intel_npu_acceleration


# Custom _torch_rmsnorm for reference, matching the one in __init__.py
def _torch_rmsnorm(input, weight, eps):
    # This is a simplified RMSNorm, might need adjustment for full compatibility
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return input * weight


class TestIntelNPULibLLM:
    @pytest.mark.parametrize(
        ("npu_op", "torch_op", "kwargs"),
        [
            (intel_npu_acceleration.relu, torch.relu, {}),
            (intel_npu_acceleration.gelu, F.gelu, {}),
            (intel_npu_acceleration.softmax, F.softmax, {"dim": -1}),
            (intel_npu_acceleration.silu, F.silu, {}),
        ],
    )
    def test_activation_ops(self, npu_op, torch_op, kwargs):
        x = torch.randn(10, 10)
        res = npu_op(x, **kwargs)
        expected = torch_op(x, **kwargs)
        assert torch.allclose(res, expected, rtol=1e-2, atol=1e-2)

    def test_silu_op(self):
        x = torch.randn(5, 7, 128)
        res = intel_npu_acceleration.silu(x)
        expected = F.silu(x)
        assert torch.allclose(res, expected, rtol=1e-2, atol=1e-2)

    def test_math_ops(self):
        class MathModel(nn.Module):
            def forward(self, x):
                return -x, torch.sin(x), torch.cos(x)

        model = MathModel()
        model.eval()
        x = torch.randn(4, 4)

        try:
            compiled = intel_npu_acceleration.compile_to_npu(model, x)
        except Exception as e:
            pytest.fail(f"Math Ops Compilation Failed: {e}")

        res_neg, res_sin, res_cos = compiled(x)

        assert torch.allclose(res_neg, -x, rtol=1e-2, atol=1e-2)
        assert torch.allclose(res_sin, torch.sin(x), rtol=1e-2, atol=1e-2)
        assert torch.allclose(res_cos, torch.cos(x), rtol=1e-2, atol=1e-2)

    def test_indexing_ops(self):
        class IndexingModel(nn.Module):
            def forward(self, x, idx, cond):
                w = torch.where(cond, x, -x)
                s = torch.index_select(x, 1, idx)
                return w, s

        model = IndexingModel()
        model.eval()

        x = torch.randn(4, 10)
        idx = torch.tensor([0, 2, 4], dtype=torch.int64)
        cond = torch.tensor([[True, False] * 5]).bool()  # (1, 10) broadcast to (4, 10)

        compiled = intel_npu_acceleration.compile_to_npu(model, (x, idx, cond))
        res_w, res_s = compiled(x, idx, cond)
        exp_w, exp_s = model(x, idx, cond)

        assert torch.allclose(res_w, exp_w, rtol=1e-2, atol=1e-2)
        assert torch.allclose(res_s, exp_s, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        ("shape", "eps", "weight_fn"),
        [
            ((2, 4, 8), 1e-6, torch.ones),
            ((1, 10, 256), 1e-5, torch.randn),
            ((3, 5, 64), 1e-3, lambda n: torch.rand(n) + 0.1),
        ],
    )
    def test_rmsnorm_op(self, shape, eps, weight_fn):
        dim = shape[-1]
        weight = weight_fn(dim)
        input = torch.randn(*shape)
        npu_res = intel_npu_acceleration.rmsnorm(input, weight, eps)
        cpu_res = _torch_rmsnorm(input, weight, eps)
        assert torch.allclose(npu_res, cpu_res, rtol=1e-2, atol=1e-2)

    def test_transpose_op(self):
        # Deterministic input
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        # Swap dim 1 and 2 -> (2, 4, 3)
        res = intel_npu_acceleration.transpose(x, 1, 2)
        expected = torch.transpose(x, 1, 2)

        assert torch.allclose(res, expected)
        assert res.shape == expected.shape

    def test_reshape_op(self):
        x = torch.randn(3, 4, 5)
        res = intel_npu_acceleration.reshape(x, (2, 30))  # Corrected shape
        expected = x.reshape(2, 30)
        assert torch.allclose(res, expected, atol=1e-3, rtol=1e-4)  # Relaxed atol

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

        x = torch.randn(1, 8, d_model)  # Batch 1, Seq 8, Dim 16

        compiled_model = intel_npu_acceleration.compile_to_npu(model, x)

        # Warmup
        _ = compiled_model(x)

        start = time.time()
        out_npu = compiled_model(x)
        print(f"Transformer Block Time: {time.time() - start:.4f}s")

        out_cpu = model(x)

        # The composite error might be higher, checking
        diff = (out_npu - out_cpu).abs().max().item()
        print(f"Transformer Block Max Diff: {diff}")
        assert torch.allclose(out_npu, out_cpu, rtol=1e-1, atol=1e-1)

    def test_rotary_embedding(self):
        """Verify fused RoPE operator against PyTorch baseline."""
        bs, slen, n_heads, head_dim = 2, 8, 4, 16
        x = torch.randn(bs, slen, n_heads, head_dim)
        cos = torch.randn(1, slen, 1, head_dim // 2)
        sin = torch.randn(1, slen, 1, head_dim // 2)

        # PyTorch reference
        d = head_dim
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        expected = torch.cat([out1, out2], dim=-1)

        res = intel_npu_acceleration.rotary_embedding(x, cos, sin)
        assert torch.allclose(res, expected, rtol=1e-2, atol=1e-2)

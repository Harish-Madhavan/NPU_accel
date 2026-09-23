"""Elementwise NPU kernels (sin/cos/exp/sqrt/abs/rsqrt/pow/clamp/where/triu)
plus max-pool autograd and flatten — forward and backward parity.

Two tiers (mirroring test_npu_backward_kernels.py):
- `TestNPUElementwiseKernels` calls the native ``_C`` kernels directly and is
  skipped without NPU hardware.
- `TestElementwiseParity` exercises the public autograd API everywhere.
"""

import pytest
import torch
from torch.testing import assert_close

import intel_npu_acceleration as npu


@pytest.mark.npu
class TestNPUElementwiseKernels:
    """Direct native-kernel verification (no fallback allowed)."""

    def test_unary_forwards(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(4, 8)
        cases = [
            (_C.npu_sin, torch.sin, x),
            (_C.npu_cos, torch.cos, x),
            (_C.npu_exp, torch.exp, x),
            (_C.npu_sqrt, torch.sqrt, x.abs() + 0.5),
            (_C.npu_abs, torch.abs, x),
            (_C.npu_sign, torch.sign, x),
            (_C.npu_rsqrt, torch.rsqrt, x.abs() + 0.5),
        ]
        for npu_fn, torch_fn, inp in cases:
            assert_close(npu_fn(inp), torch_fn(inp), atol=1e-4, rtol=1e-4)

    def test_binary_select_forwards(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(4, 8)
        assert_close(_C.npu_pow(x.abs() + 0.5, torch.full((), 2.0)),
                     torch.pow(x.abs() + 0.5, 2.0), atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_clamp(x, -1.0, 1.0), torch.clamp(x, -1.0, 1.0),
                     atol=1e-5, rtol=1e-5)
        mask = torch.randint(0, 2, (4, 8)).bool()
        assert_close(_C.npu_where(mask, x, -x), torch.where(mask, x, -x),
                     atol=1e-5, rtol=1e-5)
        assert_close(_C.npu_triu(x, 1), torch.triu(x, 1), atol=1e-5, rtol=1e-5)

    def test_unary_backwards(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(4, 8)
        go = torch.randn(4, 8)
        xp = x.abs() + 0.5
        assert_close(_C.npu_sin_backward(go, x), go * torch.cos(x), atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_cos_backward(go, x), -go * torch.sin(x), atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_exp_backward(go, x), go * torch.exp(x), atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_sqrt_backward(go, xp), go * 0.5 / torch.sqrt(xp),
                     atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_abs_backward(go, x), go * torch.sign(x), atol=1e-5, rtol=1e-5)
        assert_close(_C.npu_rsqrt_backward(go, xp), go * -0.5 * torch.pow(xp, -1.5),
                     atol=1e-4, rtol=1e-4)

    def test_binary_select_backwards(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(4, 8)
        go = torch.randn(4, 8)
        xp = x.abs() + 0.5
        assert_close(_C.npu_pow_backward(go, xp, torch.full((), 2.0)),
                     go * 2.0 * xp, atol=1e-4, rtol=1e-4)
        assert_close(_C.npu_clamp_backward(go, x, -1.0, 1.0),
                     go * ((x >= -1.0) & (x <= 1.0)).to(go.dtype), atol=1e-5, rtol=1e-5)
        mask = torch.randint(0, 2, (4, 8)).bool()
        ga, gb = _C.npu_where_backward(go, mask)
        assert_close(ga, go * mask.to(go.dtype), atol=1e-5, rtol=1e-5)
        assert_close(gb, go * (1.0 - mask.to(go.dtype)), atol=1e-5, rtol=1e-5)
        assert_close(_C.npu_triu_backward(go, 1), torch.triu(go, 1), atol=1e-5, rtol=1e-5)

    def test_max_pool2d_backward_kernel(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(2, 3, 8, 8)
        with torch.enable_grad():
            xc = x.clone().requires_grad_(True)
            out_c, _ = torch.nn.functional.max_pool2d_with_indices(xc, 2, 2)
            go = torch.randn_like(out_c)
            out_c.backward(go)
        gi = _C.npu_max_pool2d_backward(go, x, [2, 2], [2, 2], [0, 0])
        assert_close(gi, xc.grad, atol=1e-4, rtol=1e-4)


class TestElementwiseParity:
    """Public autograd API parity (NPU kernel with CPU fallback)."""

    @pytest.mark.parametrize(
        ("npu_op", "torch_op", "make_input"),
        [
            ("sin", torch.sin, lambda: torch.randn(4, 8)),
            ("cos", torch.cos, lambda: torch.randn(4, 8)),
            ("exp", torch.exp, lambda: torch.randn(4, 8) / 4),
            ("sqrt", torch.sqrt, lambda: torch.randn(4, 8).abs() + 0.5),
            ("abs", torch.abs, lambda: torch.randn(4, 8)),
            ("rsqrt", torch.rsqrt, lambda: torch.randn(4, 8).abs() + 0.5),
        ],
    )
    def test_unary_forward_backward(self, npu_op, torch_op, make_input):
        npu_fn = getattr(npu, npu_op)
        x1 = make_input().requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        out_cpu = torch_op(x1)
        out_npu = npu_fn(x2)
        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)
        go = torch.randn_like(out_cpu)
        out_cpu.backward(go)
        out_npu.backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)

    def test_pow_forward_backward(self):
        x1 = (torch.randn(4, 8).abs() + 0.5).requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        assert_close(npu.pow(x2, 2.0), torch.pow(x1, 2.0), atol=1e-3, rtol=1e-3)
        go = torch.randn(4, 8)
        torch.pow(x1, 2.0).backward(go)
        npu.pow(x2, 2.0).backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)

    def test_pow_tensor_exponent(self):
        # Tensor-exponent gradients route to the CPU fallback by design.
        x1 = (torch.randn(4, 8).abs() + 0.5).requires_grad_(True)
        b1 = torch.randn(4, 8).requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        torch.pow(x1, b1).backward(torch.ones(4, 8))
        npu.pow(x2, b2).backward(torch.ones(4, 8))
        assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
        assert_close(b2.grad, b1.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "kwargs",
        [{"min": -1.0, "max": 1.0}, {"min": 0.0}, {"max": 0.5}],
    )
    def test_clamp_forward_backward(self, kwargs):
        x1 = torch.randn(4, 8, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        assert_close(npu.clamp(x2, **kwargs), torch.clamp(x1, **kwargs), atol=1e-4, rtol=1e-4)
        go = torch.randn(4, 8)
        torch.clamp(x1, **kwargs).backward(go)
        npu.clamp(x2, **kwargs).backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-4, rtol=1e-4)

    def test_clamp_no_bounds_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            npu.clamp(torch.randn(4, 4))

    def test_where_forward_backward(self):
        mask = torch.randint(0, 2, (4, 8)).bool()
        a1 = torch.randn(4, 8, requires_grad=True)
        b1 = torch.randn(4, 8, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        assert_close(npu.where(mask, a2, b2), torch.where(mask, a1, b1), atol=1e-4, rtol=1e-4)
        go = torch.randn(4, 8)
        torch.where(mask, a1, b1).backward(go)
        npu.where(mask, a2, b2).backward(go)
        assert_close(a2.grad, a1.grad, atol=1e-4, rtol=1e-4)
        assert_close(b2.grad, b1.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("diagonal", [-1, 0, 2])
    def test_triu_forward_backward(self, diagonal):
        x1 = torch.randn(6, 8, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        assert_close(npu.triu(x2, diagonal), torch.triu(x1, diagonal), atol=1e-5, rtol=1e-5)
        go = torch.randn(6, 8)
        torch.triu(x1, diagonal).backward(go)
        npu.triu(x2, diagonal).backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-5, rtol=1e-5)

    def test_flatten(self):
        x1 = torch.randn(2, 3, 4, 5, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        assert npu.flatten(x2, 1, 2).shape == (2, 12, 5)
        assert_close(npu.flatten(x2, 1, 2), torch.flatten(x1, 1, 2), atol=1e-5, rtol=1e-5)
        go = torch.randn(2, 12, 5)
        torch.flatten(x1, 1, 2).backward(go)
        npu.flatten(x2, 1, 2).backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-5, rtol=1e-5)

    def test_max_pool2d_training_step(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.MaxPool2d(2))
        ref = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.MaxPool2d(2))
        ref.load_state_dict(model.state_dict())
        x = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 8, 8, 8)

        h = npu.conv2d(x, model[0].weight, model[0].bias, (1, 1), (1, 1), (1, 1), 1)
        out = npu.max_pool2d(h, 2, 2)
        loss = torch.nn.functional.mse_loss(out, target)
        model.zero_grad()
        loss.backward()

        out_ref = ref(x)
        loss_ref = torch.nn.functional.mse_loss(out_ref, target)
        loss_ref.backward()
        assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
        assert_close(model[0].weight.grad, ref[0].weight.grad, atol=1e-2, rtol=1e-2)

    def test_max_pool2d_overlapping_fallback(self):
        # Overlapping windows need scatter-add: must still be correct via CPU fallback.
        x1 = torch.randn(1, 2, 7, 7, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        out_npu = npu.max_pool2d(x2, 3, 2, 1)
        out_ref = torch.nn.functional.max_pool2d(x1, 3, 2, 1)
        assert_close(out_npu, out_ref, atol=1e-3, rtol=1e-3)
        go = torch.randn_like(out_ref)
        out_ref.backward(go)
        out_npu.backward(go)
        assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)

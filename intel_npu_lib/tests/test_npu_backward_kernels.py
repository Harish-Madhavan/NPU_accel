"""Level Zero backward-kernel tests — every NPU autograd path executes on device.

Two tiers:
- `TestNPUBackwardKernels` calls the native ``_C`` kernels directly and is
  skipped without NPU hardware (a fallback would mask a broken kernel here).
- `TestBackwardParity` exercises the public autograd API (NPU kernel with CPU
  fallback) and runs everywhere, including hardware-less CI.
"""

import pytest
import torch
from torch.testing import assert_close

import intel_npu_acceleration as npu


@pytest.mark.npu
class TestNPUBackwardKernels:
    """Direct native-kernel verification (no fallback allowed)."""

    def test_rmsnorm_backward_kernel(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        x = torch.randn(2, 16)
        w = torch.randn(16)
        go = torch.randn(2, 16)
        gi, gw = _C.npu_rmsnorm_backward(go, x, w, 1e-6)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        xn = x / rms
        assert_close(gw, (go * xn).sum(0), atol=1e-4, rtol=1e-4)
        dl = go * w
        assert_close(gi, (dl - xn * (dl * xn).mean(-1, keepdim=True)) / rms, atol=1e-4, rtol=1e-4)

    def test_embedding_backward_kernel(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        go = torch.randn(4, 8)
        idx = torch.tensor([1, 3, 1, 7])
        gw = _C.npu_embedding_backward(go, idx, 10)
        assert_close(gw, torch.zeros(10, 8).index_add_(0, idx, go), atol=1e-5, rtol=1e-5)

    def test_mse_loss_backward_kernel(self, skip_if_no_npu):
        from intel_npu_acceleration import _C

        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        go = torch.randn(4, 8)
        for reduction, code in [("mean", 1), ("sum", 2), ("none", 0)]:
            scale = 2.0 / pred.numel() if reduction == "mean" else 2.0
            assert_close(
                _C.npu_mse_loss_backward(go, pred, target, code),
                go * scale * (pred - target),
                atol=1e-5,
                rtol=1e-5,
            )

    @pytest.mark.parametrize(
        "stride, padding, groups",
        [(1, 1, 1), (2, 1, 1), (1, 0, 1), (1, 1, 2)],
    )
    def test_conv2d_backward_kernel(self, skip_if_no_npu, stride, padding, groups):
        from intel_npu_acceleration import _C

        x = torch.randn(2, 4, 8, 8)
        w = torch.randn(6, 4 // groups, 3, 3)
        out = torch.nn.functional.conv2d(x, w, None, stride=stride, padding=padding, groups=groups)
        go = torch.randn_like(out)
        gi, gw, gb = _C.npu_conv2d_backward(
            go, x, w, [stride, stride], [padding, padding], [1, 1], groups, True, True, True
        )
        assert go.shape == out.shape
        assert_close(
            gi,
            torch.nn.grad.conv2d_input(x.shape, w, go, stride=stride, padding=padding, groups=groups),
            atol=1e-3,
            rtol=1e-3,
        )
        assert_close(
            gw,
            torch.nn.grad.conv2d_weight(x, w.shape, go, stride=stride, padding=padding, groups=groups),
            atol=1e-3,
            rtol=1e-3,
        )
        assert_close(gb, go.sum(dim=(0, 2, 3)), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("causal", [False, True])
    def test_sdpa_backward_kernel(self, skip_if_no_npu, causal):
        from intel_npu_acceleration import _C

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        go = torch.randn_like(out)
        gq, gk, gv = _C.npu_scaled_dot_product_attention_backward(
            go, q, k, v, torch.empty(0), 0.0, causal, 0.0
        )
        q1 = q.clone().requires_grad_(True)
        k1 = k.clone().requires_grad_(True)
        v1 = v.clone().requires_grad_(True)
        torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, is_causal=causal).backward(go)
        assert_close(gq, q1.grad, atol=1e-3, rtol=1e-3)
        assert_close(gk, k1.grad, atol=1e-3, rtol=1e-3)
        assert_close(gv, v1.grad, atol=1e-3, rtol=1e-3)


class TestBackwardParity:
    """Public autograd API parity (NPU kernel with CPU fallback)."""

    def test_conv2d_training_step(self):
        model = torch.nn.Conv2d(3, 8, 3, padding=1)
        x = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 8, 16, 16)

        ref = torch.nn.Conv2d(3, 8, 3, padding=1)
        ref.load_state_dict(model.state_dict())

        out = npu.conv2d(x, model.weight, model.bias, (1, 1), (1, 1), (1, 1), 1)
        loss = torch.nn.functional.mse_loss(out, target)
        model.zero_grad()
        loss.backward()
        gw_npu, gb_npu = model.weight.grad.clone(), model.bias.grad.clone()

        out_ref = ref(x)
        loss_ref = torch.nn.functional.mse_loss(out_ref, target)
        loss_ref.backward()
        assert_close(gw_npu, ref.weight.grad, atol=1e-2, rtol=1e-2)
        assert_close(gb_npu, ref.bias.grad, atol=1e-2, rtol=1e-2)

    def test_mse_loss_module_backward(self):
        criterion = npu.nn.MSELoss()
        pred = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)
        loss_npu = criterion(pred, target)
        loss_cpu = torch.nn.functional.mse_loss(pred.detach(), target)
        assert_close(loss_npu, loss_cpu, atol=1e-4, rtol=1e-4)
        loss_npu.backward()
        assert pred.grad is not None

    def test_embedding_module_backward(self):
        model = npu.nn.Embedding(20, 16)
        ref = torch.nn.Embedding(20, 16)
        ref.load_state_dict(model.state_dict())
        idx = torch.randint(0, 20, (2, 8))
        out_npu = model(idx)
        out_ref = ref(idx)
        assert_close(out_npu, out_ref, atol=1e-3, rtol=1e-3)
        out_npu.sum().backward()
        out_ref.sum().backward()
        assert_close(model.weight.grad, ref.weight.grad, atol=1e-3, rtol=1e-3)

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.testing import assert_close

import intel_npu_acceleration as npu
import intel_npu_acceleration.functional as F_npu


class TestPyTorchConformance:
    """
    Test PyTorch API conformance, tensor layout compatibility,
    numerical precision, non-contiguous striding, and autograd gradient equivalence.
    """

    @pytest.mark.parametrize(
        ("npu_op", "torch_op", "is_binary"),
        [
            (F_npu.add, torch.add, True),
            (F_npu.sub, torch.sub, True),
            (F_npu.mul, torch.mul, True),
            (F_npu.div, torch.div, True),
            (F_npu.neg, torch.neg, False),
        ],
    )
    def test_elementwise_autograd_conformance(self, npu_op, torch_op, is_binary):
        """Verify autograd forward and backward against CPU PyTorch for elementwise ops."""
        shape = (4, 8)
        a_cpu = torch.randn(shape, requires_grad=True)
        a_npu = a_cpu.detach().clone().requires_grad_(True)

        if is_binary:
            # Use safe non-zero values for division
            b_cpu = torch.randn(shape).abs() + 0.5
            b_cpu.requires_grad_(True)
            b_npu = b_cpu.detach().clone().requires_grad_(True)

            out_cpu = torch_op(a_cpu, b_cpu)
            out_npu = npu_op(a_npu, b_npu)
        else:
            out_cpu = torch_op(a_cpu)
            out_npu = npu_op(a_npu)

        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        out_npu.backward(grad)

        assert_close(a_npu.grad, a_cpu.grad, atol=1e-2, rtol=1e-2)
        if is_binary:
            assert_close(b_npu.grad, b_cpu.grad, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        ("npu_act", "torch_act"),
        [
            (F_npu.relu, nn.functional.relu),
            (F_npu.gelu, nn.functional.gelu),
            (F_npu.silu, nn.functional.silu),
            (F_npu.hardsigmoid, nn.functional.hardsigmoid),
            (F_npu.hardswish, nn.functional.hardswish),
        ],
    )
    def test_activation_autograd_conformance(self, npu_act, torch_act):
        """Verify activations against PyTorch native F implementations."""
        x1 = torch.randn(3, 12, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = torch_act(x1)
        out_npu = npu_act(x2)

        assert_close(out_npu, out_cpu, atol=1e-2, rtol=1e-2)

        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        out_npu.backward(grad)

        assert_close(x2.grad, x1.grad, atol=1e-2, rtol=1e-2)

    def test_non_contiguous_tensor_inputs(self):
        """Verify that sliced / transposed non-contiguous tensors execute correctly."""
        base = torch.randn(8, 16)
        x_non_contiguous = base[:, ::2]  # stride is (16, 2)
        assert not x_non_contiguous.is_contiguous()

        linear = nn.Linear(8, 4)
        linear.eval()

        compiled_linear = npu.compile(linear, x_non_contiguous)
        out_compiled = compiled_linear(x_non_contiguous)
        out_expected = linear(x_non_contiguous)

        assert_close(out_compiled, out_expected, atol=1e-2, rtol=1e-2)

    def test_end_to_end_training_step(self):
        """Verify full PyTorch training loop step with forward, backward, and optimizer update."""
        # Model on PyTorch CPU
        model_cpu = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        # Model for NPU accelerated forward
        model_npu = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        model_npu.load_state_dict(model_cpu.state_dict())

        opt_cpu = optim.AdamW(model_cpu.parameters(), lr=1e-2)
        opt_npu = optim.AdamW(model_npu.parameters(), lr=1e-2)

        x = torch.randn(4, 8)
        target = torch.randn(4, 4)

        # CPU step
        opt_cpu.zero_grad()
        out_c = model_cpu(x)
        loss_c = nn.functional.mse_loss(out_c, target)
        loss_c.backward()
        opt_cpu.step()

        # NPU step
        opt_npu.zero_grad()
        # Use functional autograd ops
        h1 = F_npu.relu(F_npu.linear(x, model_npu[0].weight, model_npu[0].bias))
        out_n = F_npu.linear(h1, model_npu[2].weight, model_npu[2].bias)
        loss_n = nn.functional.mse_loss(out_n, target)
        loss_n.backward()
        opt_npu.step()

        # Verify loss and updated weights match
        assert_close(loss_n, loss_c, atol=1e-2, rtol=1e-2)
        for p_cpu, p_npu in zip(model_cpu.parameters(), model_npu.parameters(), strict=False):
            assert_close(p_npu, p_cpu, atol=1e-2, rtol=1e-2)

    def test_torch_compile_modes_conformance(self):
        """Verify that standard PyTorch 2.x compile modes (reduce-overhead, max-autotune, dynamic) work seamlessly."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        x = torch.randn(2, 16)
        expected = model(x)

        # 1. Standard mode
        opt_std = torch.compile(model, backend="npu")
        assert_close(opt_std(x), expected, atol=1e-2, rtol=1e-2)

        # 2. reduce-overhead mode
        opt_red = torch.compile(model, backend="npu", mode="reduce-overhead")
        assert_close(opt_red(x), expected, atol=1e-2, rtol=1e-2)

        # 3. intel_npu alias
        opt_alias = torch.compile(model, backend="intel_npu")
        assert_close(opt_alias(x), expected, atol=1e-2, rtol=1e-2)

        # 4. dynamic shapes mode
        opt_dyn = torch.compile(model, backend="npu", dynamic=True)
        x_dyn = torch.randn(5, 16)
        assert_close(opt_dyn(x_dyn), model(x_dyn), atol=1e-2, rtol=1e-2)

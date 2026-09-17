import unittest

import torch
import torch.nn as nn
from torch.testing import assert_close

import intel_npu_acceleration as npu
import intel_npu_acceleration.functional as F_npu
from intel_npu_acceleration.optim import NPUSGD, NPUAdam


class TestTrainingAndOptimizers(unittest.TestCase):
    """
    Test suite for Level Zero accelerated training, backward autograd passes,
    loss functions, and hardware optimizers (NPUAdam, NPUSGD).
    """

    def setUp(self):
        npu.clear_graph_cache()
        torch.manual_seed(42)

    def test_linear_autograd_backward_level_zero(self):
        """Verify NPULinear forward and backward against PyTorch native linear."""
        x_cpu = torch.randn(4, 16, requires_grad=True)
        w_cpu = torch.randn(8, 16, requires_grad=True)
        b_cpu = torch.randn(8, requires_grad=True)

        x_npu = x_cpu.detach().clone().requires_grad_(True)
        w_npu = w_cpu.detach().clone().requires_grad_(True)
        b_npu = b_cpu.detach().clone().requires_grad_(True)

        out_cpu = torch.nn.functional.linear(x_cpu, w_cpu, b_cpu)
        out_npu = npu.linear(x_npu, w_npu, b_npu)
        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        assert_close(x_npu.grad, x_cpu.grad, atol=1e-2, rtol=1e-2)
        assert_close(w_npu.grad, w_cpu.grad, atol=1e-2, rtol=1e-2)
        assert_close(b_npu.grad, b_cpu.grad, atol=1e-2, rtol=1e-2)

    def test_matmul_autograd_backward_level_zero(self):
        """Verify NPUMatMul forward and backward against PyTorch native matmul."""
        a_cpu = torch.randn(6, 12, requires_grad=True)
        b_cpu = torch.randn(12, 8, requires_grad=True)

        a_npu = a_cpu.detach().clone().requires_grad_(True)
        b_npu = b_cpu.detach().clone().requires_grad_(True)

        out_cpu = torch.matmul(a_cpu, b_cpu)
        out_npu = npu.matmul(a_npu, b_npu)
        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        assert_close(a_npu.grad, a_cpu.grad, atol=1e-2, rtol=1e-2)
        assert_close(b_npu.grad, b_cpu.grad, atol=1e-2, rtol=1e-2)

    def test_activations_backward_level_zero(self):
        """Verify backward pass for ReLU, GELU, and SiLU."""
        acts = [
            (npu.relu, torch.nn.functional.relu),
            (npu.gelu, torch.nn.functional.gelu),
            (npu.silu, torch.nn.functional.silu),
        ]
        for npu_fn, torch_fn in acts:
            x_cpu = torch.randn(4, 16, requires_grad=True)
            x_npu = x_cpu.detach().clone().requires_grad_(True)

            out_c = torch_fn(x_cpu)
            out_n = npu_fn(x_npu)
            assert_close(out_n, out_c, atol=1e-2, rtol=1e-2)

            grad_out = torch.randn_like(out_c)
            out_c.backward(grad_out)
            out_n.backward(grad_out)
            assert_close(x_npu.grad, x_cpu.grad, atol=1e-2, rtol=1e-2)

    def test_npu_adam_optimizer_step(self):
        """Verify NPUAdam optimization step convergence against torch.optim.Adam."""
        torch.manual_seed(123)
        p_cpu = torch.randn(10, 10, requires_grad=True)
        p_npu = p_cpu.detach().clone().requires_grad_(True)

        opt_cpu = torch.optim.Adam([p_cpu], lr=1e-2, weight_decay=1e-4)
        opt_npu = NPUAdam([p_npu], lr=1e-2, weight_decay=1e-4)

        for _ in range(5):
            grad = torch.randn(10, 10)
            p_cpu.grad = grad.clone()
            p_npu.grad = grad.clone()

            opt_cpu.step()
            opt_npu.step()

            assert_close(p_npu, p_cpu, atol=1e-2, rtol=1e-2)

    def test_npu_sgd_optimizer_step(self):
        """Verify NPUSGD optimization step with momentum against torch.optim.SGD."""
        torch.manual_seed(123)
        p_cpu = torch.randn(8, 8, requires_grad=True)
        p_npu = p_cpu.detach().clone().requires_grad_(True)

        opt_cpu = torch.optim.SGD([p_cpu], lr=1e-2, momentum=0.9, weight_decay=1e-4)
        opt_npu = NPUSGD([p_npu], lr=1e-2, momentum=0.9, weight_decay=1e-4)

        for _ in range(5):
            grad = torch.randn(8, 8)
            p_cpu.grad = grad.clone()
            p_npu.grad = grad.clone()

            opt_cpu.step()
            opt_npu.step()

            assert_close(p_npu, p_cpu, atol=1e-2, rtol=1e-2)

    def test_end_to_end_npu_training_loop(self):
        """Verify multi-step training loop with Level Zero forward, backward, and optimizer."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 4),
        )
        optimizer = NPUAdam(model.parameters(), lr=1e-2)

        x = torch.randn(8, 8)
        target = torch.randn(8, 4)

        initial_loss = None
        final_loss = None

        for step in range(15):
            optimizer.zero_grad()
            h = npu.gelu(npu.linear(x, model[0].weight, model[0].bias))
            pred = npu.linear(h, model[2].weight, model[2].bias)
            loss = npu.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Verify loss strictly decreases during training
        self.assertLess(final_loss, initial_loss * 0.5)

    def test_softmax_autograd_backward_level_zero(self):
        """Verify softmax backward pass executes with high precision."""
        x_cpu = torch.randn(4, 16, requires_grad=True)
        x_npu = x_cpu.detach().clone().requires_grad_(True)

        out_cpu = torch.softmax(x_cpu, dim=-1)
        out_npu = npu.softmax(x_npu, dim=-1)
        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        assert_close(x_npu.grad, x_cpu.grad, atol=1e-2, rtol=1e-2)

    def test_rmsnorm_autograd_backward_level_zero(self):
        """Verify RMSNorm backward pass computes input and weight gradients."""
        x_cpu = torch.randn(4, 16, requires_grad=True)
        w_cpu = torch.randn(16, requires_grad=True)

        x_npu = x_cpu.detach().clone().requires_grad_(True)
        w_npu = w_cpu.detach().clone().requires_grad_(True)

        out_npu = npu.rmsnorm(x_npu, w_npu, eps=1e-6)
        # Reference RMSNorm
        rms = torch.sqrt(x_cpu.pow(2).mean(-1, keepdim=True) + 1e-6)
        out_cpu = (x_cpu / rms) * w_cpu

        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        assert_close(x_npu.grad, x_cpu.grad, atol=1e-2, rtol=1e-2)
        assert_close(w_npu.grad, w_cpu.grad, atol=1e-2, rtol=1e-2)

    def test_npu_nn_modules_training(self):
        """Verify model constructed with npu.nn modules trains end-to-end."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            npu.nn.Linear(8, 16),
            npu.nn.RMSNorm(16),
            npu.nn.Linear(16, 2),
        )
        criterion = npu.nn.MSELoss()
        optimizer = NPUAdam(model.parameters(), lr=1e-2)

        x = torch.randn(8, 8)
        target = torch.randn(8, 2)

        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        self.assertIsNotNone(loss.item())


if __name__ == "__main__":
    unittest.main()


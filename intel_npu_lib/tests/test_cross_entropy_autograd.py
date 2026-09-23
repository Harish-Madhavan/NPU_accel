import torch
from torch.testing import assert_close

import intel_npu_acceleration as npu
from intel_npu_acceleration.optim import clip_grad_norm_


class TestCrossEntropyAutograd:
    def test_cross_entropy_forward_backward_indices(self):
        batch_size = 4
        num_classes = 10
        pred_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        pred_npu = pred_cpu.detach().clone().requires_grad_(True)

        # Forward
        loss_cpu = torch.nn.functional.cross_entropy(pred_cpu, target, reduction="mean")
        loss_npu = npu.cross_entropy_loss(pred_npu, target, reduction="mean")

        assert_close(loss_npu, loss_cpu, atol=1e-3, rtol=1e-3)

        # Backward
        loss_cpu.backward()
        loss_npu.backward()

        assert_close(pred_npu.grad, pred_cpu.grad, atol=1e-3, rtol=1e-3)

    def test_cross_entropy_forward_backward_probabilities(self):
        batch_size = 4
        num_classes = 5
        pred_cpu = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.softmax(torch.randn(batch_size, num_classes), dim=-1)

        pred_npu = pred_cpu.detach().clone().requires_grad_(True)

        loss_cpu = torch.nn.functional.cross_entropy(pred_cpu, target, reduction="mean")
        loss_npu = npu.cross_entropy_loss(pred_npu, target, reduction="mean")

        assert_close(loss_npu, loss_cpu, atol=1e-3, rtol=1e-3)

        loss_cpu.backward()
        loss_npu.backward()

        assert_close(pred_npu.grad, pred_cpu.grad, atol=1e-3, rtol=1e-3)

    def test_cross_entropy_reductions(self):
        batch_size = 3
        num_classes = 4
        pred = torch.randn(batch_size, num_classes, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        # Sum reduction
        pred_c = pred.detach().clone().requires_grad_(True)
        pred_n = pred.detach().clone().requires_grad_(True)
        loss_c = torch.nn.functional.cross_entropy(pred_c, target, reduction="sum")
        loss_n = npu.cross_entropy_loss(pred_n, target, reduction="sum")
        assert_close(loss_n, loss_c, atol=1e-3, rtol=1e-3)

        loss_c.backward()
        loss_n.backward()
        assert_close(pred_n.grad, pred_c.grad, atol=1e-3, rtol=1e-3)

        # None reduction
        loss_none_c = torch.nn.functional.cross_entropy(pred, target, reduction="none")
        loss_none_n = npu.cross_entropy_loss(pred, target, reduction="none")
        assert_close(loss_none_n, loss_none_c, atol=1e-3, rtol=1e-3)

    def test_cross_entropy_module(self):
        criterion_cpu = torch.nn.CrossEntropyLoss()
        criterion_npu = npu.nn.CrossEntropyLoss()

        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_npu = x_cpu.detach().clone().requires_grad_(True)
        target = torch.tensor([1, 0, 7, 3])

        out_cpu = criterion_cpu(x_cpu, target)
        out_npu = criterion_npu(x_npu, target)

        assert_close(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        out_cpu.backward()
        out_npu.backward()
        assert_close(x_npu.grad, x_cpu.grad, atol=1e-3, rtol=1e-3)

    def test_clip_grad_norm(self):
        p1_cpu = torch.randn(10, 10, requires_grad=True)
        p2_cpu = torch.randn(10, requires_grad=True)
        loss_cpu = (p1_cpu * 2.0).sum() + (p2_cpu * 3.0).sum()
        loss_cpu.backward()

        p1_npu = p1_cpu.detach().clone()
        p1_npu.grad = p1_cpu.grad.detach().clone()
        p2_npu = p2_cpu.detach().clone()
        p2_npu.grad = p2_cpu.grad.detach().clone()

        norm_cpu = torch.nn.utils.clip_grad_norm_([p1_cpu, p2_cpu], max_norm=1.0)
        norm_npu = clip_grad_norm_([p1_npu, p2_npu], max_norm=1.0)

        assert_close(norm_npu, norm_cpu, atol=1e-4, rtol=1e-4)
        assert_close(p1_npu.grad, p1_cpu.grad, atol=1e-4, rtol=1e-4)
        assert_close(p2_npu.grad, p2_cpu.grad, atol=1e-4, rtol=1e-4)

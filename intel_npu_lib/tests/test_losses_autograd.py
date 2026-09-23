import torch
from torch.testing import assert_close

import intel_npu_acceleration as npu


class TestLossesAutograd:
    def test_l1_loss_forward_backward_mean(self):
        pred_cpu = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)
        pred_npu = pred_cpu.detach().clone().requires_grad_(True)

        loss_cpu = torch.nn.functional.l1_loss(pred_cpu, target, reduction="mean")
        loss_npu = npu.l1_loss(pred_npu, target, reduction="mean")

        assert_close(loss_npu, loss_cpu, atol=1e-4, rtol=1e-4)

        loss_cpu.backward()
        loss_npu.backward()

        assert_close(pred_npu.grad, pred_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_l1_loss_reductions(self):
        pred = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)

        # sum reduction
        pred_c = pred.detach().clone().requires_grad_(True)
        pred_n = pred.detach().clone().requires_grad_(True)
        loss_c = torch.nn.functional.l1_loss(pred_c, target, reduction="sum")
        loss_n = npu.l1_loss(pred_n, target, reduction="sum")
        assert_close(loss_n, loss_c, atol=1e-4, rtol=1e-4)

        loss_c.backward()
        loss_n.backward()
        assert_close(pred_n.grad, pred_c.grad, atol=1e-4, rtol=1e-4)

        # none reduction
        loss_none_c = torch.nn.functional.l1_loss(pred, target, reduction="none")
        loss_none_n = npu.l1_loss(pred, target, reduction="none")
        assert_close(loss_none_n, loss_none_c, atol=1e-4, rtol=1e-4)

    def test_l1_loss_module(self):
        criterion_cpu = torch.nn.L1Loss(reduction="mean")
        criterion_npu = npu.nn.L1Loss(reduction="mean")

        x_cpu = torch.randn(2, 6, requires_grad=True)
        x_npu = x_cpu.detach().clone().requires_grad_(True)
        target = torch.randn(2, 6)

        out_cpu = criterion_cpu(x_cpu, target)
        out_npu = criterion_npu(x_npu, target)
        assert_close(out_npu, out_cpu, atol=1e-4, rtol=1e-4)

        out_cpu.backward()
        out_npu.backward()
        assert_close(x_npu.grad, x_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_bce_with_logits_loss_forward_backward_mean(self):
        pred_cpu = torch.randn(4, 5, requires_grad=True)
        target = torch.empty(4, 5).random_(2).float()
        pred_npu = pred_cpu.detach().clone().requires_grad_(True)

        loss_cpu = torch.nn.functional.binary_cross_entropy_with_logits(pred_cpu, target, reduction="mean")
        loss_npu = npu.bce_with_logits_loss(pred_npu, target, reduction="mean")

        assert_close(loss_npu, loss_cpu, atol=1e-4, rtol=1e-4)

        loss_cpu.backward()
        loss_npu.backward()

        assert_close(pred_npu.grad, pred_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_bce_with_logits_loss_with_pos_weight(self):
        pred_cpu = torch.randn(3, 4, requires_grad=True)
        target = torch.empty(3, 4).random_(2).float()
        pos_weight = torch.tensor([1.5, 2.0, 0.5, 1.0])
        pred_npu = pred_cpu.detach().clone().requires_grad_(True)

        loss_cpu = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_cpu, target, reduction="mean", pos_weight=pos_weight
        )
        loss_npu = npu.bce_with_logits_loss(
            pred_npu, target, reduction="mean", pos_weight=pos_weight
        )

        assert_close(loss_npu, loss_cpu, atol=1e-4, rtol=1e-4)

        loss_cpu.backward()
        loss_npu.backward()

        assert_close(pred_npu.grad, pred_cpu.grad, atol=1e-4, rtol=1e-4)

    def test_bce_with_logits_loss_module(self):
        criterion_cpu = torch.nn.BCEWithLogitsLoss()
        criterion_npu = npu.nn.BCEWithLogitsLoss()

        x_cpu = torch.randn(2, 4, requires_grad=True)
        x_npu = x_cpu.detach().clone().requires_grad_(True)
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

        out_cpu = criterion_cpu(x_cpu, target)
        out_npu = criterion_npu(x_npu, target)
        assert_close(out_npu, out_cpu, atol=1e-4, rtol=1e-4)

        out_cpu.backward()
        out_npu.backward()
        assert_close(x_npu.grad, x_cpu.grad, atol=1e-4, rtol=1e-4)

import unittest

import torch

import intel_npu_acceleration as npu


class TestAutogradStructural(unittest.TestCase):
    def setUp(self):
        # Clean the compiler graph cache before each test
        import intel_npu_acceleration.frontend as frontend
        frontend._GRAPH_CACHE.clear()

    def test_neg_autograd(self):
        x1 = torch.randn(2, 3, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = torch.neg(x1)
        out_npu = npu.neg(x2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_div_autograd(self):
        a1 = torch.randn(2, 3, requires_grad=True)
        b1 = (torch.rand(2, 3) + 0.5).requires_grad_(True)  # keep away from 0

        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_cpu = torch.div(a1, b1)
        out_npu = npu.div(a2, b2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(a2.grad, a1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(b2.grad, b1.grad, atol=1e-2, rtol=1e-2))

    def test_transpose_autograd(self):
        x1 = torch.randn(2, 3, 4, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = x1.transpose(1, 2)
        out_npu = npu.transpose(x2, 1, 2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_reshape_autograd(self):
        x1 = torch.randn(2, 3, 4, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = x1.reshape(2, 12)
        out_npu = npu.reshape(x2, [2, 12])

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_cat_autograd(self):
        a1 = torch.randn(2, 3, requires_grad=True)
        b1 = torch.randn(2, 4, requires_grad=True)

        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_cpu = torch.cat([a1, b1], dim=1)
        out_npu = npu.cat([a2, b2], dim=1)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(a2.grad, a1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(b2.grad, b1.grad, atol=1e-2, rtol=1e-2))

    def test_stack_autograd(self):
        a1 = torch.randn(2, 3, requires_grad=True)
        b1 = torch.randn(2, 3, requires_grad=True)

        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_cpu = torch.stack([a1, b1], dim=0)
        out_npu = npu.stack([a2, b2], dim=0)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(a2.grad, a1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(b2.grad, b1.grad, atol=1e-2, rtol=1e-2))

    def test_mean_autograd_global(self):
        x1 = torch.randn(2, 3, 4, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = torch.mean(x1)
        out_npu = npu.mean(x2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_mean_autograd_dim_keepdim(self):
        cases = [
            (1, False),
            (1, True),
            ((1, 2), False),
            ((1, 2), True),
        ]

        for dim, keepdim in cases:
            with self.subTest(dim=dim, keepdim=keepdim):
                x1 = torch.randn(2, 3, 4, requires_grad=True)
                x2 = x1.detach().clone().requires_grad_(True)

                out_cpu = torch.mean(x1, dim=dim, keepdim=keepdim)
                out_npu = npu.mean(x2, dim=dim, keepdim=keepdim)

                self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

                grad_out = torch.randn_like(out_cpu)
                out_cpu.backward(grad_out)
                out_npu.backward(grad_out)

                self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()

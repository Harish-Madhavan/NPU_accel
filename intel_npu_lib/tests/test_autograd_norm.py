import unittest
import torch
import intel_npu_acceleration as npu


class TestAutogradNormAndActivations(unittest.TestCase):
    def setUp(self):
        # Clean the compiler graph cache before each test
        import intel_npu_acceleration.frontend as frontend
        frontend._GRAPH_CACHE.clear()

    def test_layer_norm_autograd(self):
        # Test LayerNorm forward and backward
        # Setup inputs
        x_shape = (2, 3, 4)
        normalized_shape = (4,)

        x1 = torch.randn(x_shape, requires_grad=True)
        w1 = torch.randn(normalized_shape, requires_grad=True)
        b1 = torch.randn(normalized_shape, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        # CPU PyTorch
        out_cpu = torch.nn.functional.layer_norm(x1, normalized_shape, w1, b1, eps=1e-5)
        # NPU Autograd
        out_npu = npu.layer_norm(x2, normalized_shape, w2, b2, eps=1e-5)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        # Backward
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(w2.grad, w1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(b2.grad, b1.grad, atol=1e-2, rtol=1e-2))

    def test_layer_norm_autograd_no_weight_bias(self):
        # Test LayerNorm without weight and bias
        x_shape = (2, 3, 4)
        normalized_shape = (4,)

        x1 = torch.randn(x_shape, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        # CPU PyTorch
        out_cpu = torch.nn.functional.layer_norm(x1, normalized_shape, None, None, eps=1e-5)
        # NPU Autograd
        out_npu = npu.layer_norm(x2, normalized_shape, None, None, eps=1e-5)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        # Backward
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_hardsigmoid_autograd(self):
        # Test HardSigmoid forward and backward (testing boundary conditions too)
        x1 = torch.tensor([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0], requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = torch.nn.functional.hardsigmoid(x1)
        out_npu = npu.hardsigmoid(x2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.ones_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))

    def test_hardswish_autograd(self):
        # Test HardSwish forward and backward (testing boundary conditions too)
        x1 = torch.tensor([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0], requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out_cpu = torch.nn.functional.hardswish(x1)
        out_npu = npu.hardswish(x2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        grad_out = torch.ones_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(x2.grad, x1.grad, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()

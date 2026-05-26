import unittest
import torch
import intel_npu_acceleration as npu


class TestAutogradSLM(unittest.TestCase):
    def setUp(self):
        # Clean the compiler graph cache before each test
        import intel_npu_acceleration.frontend as frontend
        frontend._GRAPH_CACHE.clear()

    def test_embedding_autograd(self):
        # Setup inputs
        vocab_size = 20
        embed_dim = 16
        seq_len = 8
        batch_size = 2

        # Indices must be long integers
        indices = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Weights require grad
        w1 = torch.randn(vocab_size, embed_dim, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_(True)

        # CPU PyTorch
        out_cpu = torch.nn.functional.embedding(indices, w1)
        # NPU Autograd
        out_npu = npu.embedding(indices, w2)

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        # Backward
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(w2.grad, w1.grad, atol=1e-2, rtol=1e-2))

    def test_sdpa_autograd_causal(self):
        # Setup SDPA inputs
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16

        q1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        k1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        v1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)

        # CPU PyTorch (using causal attention)
        out_cpu = torch.nn.functional.scaled_dot_product_attention(
            q1, k1, v1, is_causal=True
        )
        # NPU Autograd (using causal attention)
        out_npu = npu.scaled_dot_product_attention(
            q2, k2, v2, is_causal=True
        )

        self.assertTrue(torch.allclose(out_npu, out_cpu, atol=1e-2, rtol=1e-2))

        # Backward
        grad_out = torch.randn_like(out_cpu)
        out_cpu.backward(grad_out)
        out_npu.backward(grad_out)

        self.assertTrue(torch.allclose(q2.grad, q1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(k2.grad, k1.grad, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(v2.grad, v1.grad, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()

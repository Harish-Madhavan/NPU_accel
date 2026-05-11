import unittest
import torch
import intel_npu_acceleration as npu


class TestParametrizedOps(unittest.TestCase):
    def test_add_shapes(self):
        shapes = [
            ((1,), (1,)),
            ((10,), (10,)),
            ((2, 2), (2, 2)),
            ((2, 3, 4), (2, 3, 4)),
            ((1, 2, 3, 4), (1, 2, 3, 4)),
        ]

        for s1, s2 in shapes:
            with self.subTest(shape=s1):
                a = torch.randn(s1)
                b = torch.randn(s2)
                res = npu.add(a, b)
                self.assertTrue(torch.allclose(res, a + b, atol=1e-2, rtol=1e-2))

    def test_matmul_shapes(self):
        # (M, K) x (K, N) -> (M, N)
        shapes = [(2, 2, 2), (10, 20, 5), (1, 10, 1)]
        for m, k, n in shapes:
            with self.subTest(m=m, k=k, n=n):
                a = torch.randn(m, k)
                b = torch.randn(k, n)
                res = npu.matmul(a, b)
                self.assertTrue(
                    torch.allclose(res, torch.matmul(a, b), atol=0.1, rtol=0.1)
                )

    def test_broadcasting(self):
        # NPU Lib handles broadcasting via OpenVINO Add/Mul
        cases = [
            ((2, 3), (3,)),  # Add vector to matrix
            ((2, 3, 4), (1, 1, 4)),
            ((2, 3, 4), (1,)),
        ]

        for s1, s2 in cases:
            with self.subTest(s1=s1, s2=s2):
                a = torch.randn(s1)
                b = torch.randn(s2)
                res = npu.add(a, b)
                self.assertTrue(torch.allclose(res, a + b, atol=1e-2, rtol=1e-2))

    def test_broadcasting_backward(self):
        cases = [
            ((2, 3), (3,)),  # Add vector to matrix
            ((2, 3, 4), (1, 1, 4)),
            ((2, 3, 4), (1,)),
        ]

        for s1, s2 in cases:
            with self.subTest(s1=s1, s2=s2):
                a1 = torch.randn(s1, requires_grad=True)
                b1 = torch.randn(s2, requires_grad=True)
                res1 = npu.add(a1, b1)
                res1.sum().backward()

                a2 = a1.detach().clone().requires_grad_(True)
                b2 = b1.detach().clone().requires_grad_(True)
                res2 = a2 + b2
                res2.sum().backward()

                self.assertTrue(torch.allclose(a1.grad, a2.grad, atol=1e-3))
                self.assertTrue(torch.allclose(b1.grad, b2.grad, atol=1e-3))


if __name__ == "__main__":
    unittest.main()

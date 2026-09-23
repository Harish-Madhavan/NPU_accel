import pytest
import torch

import intel_npu_acceleration as npu


@pytest.mark.parametrize(
    "s1, s2",
    [
        ((1,), (1,)),
        ((10,), (10,)),
        ((2, 2), (2, 2)),
        ((2, 3, 4), (2, 3, 4)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
    ],
)
def test_add_shapes(s1, s2):
    a = torch.randn(s1)
    b = torch.randn(s2)
    res = npu.add(a, b)
    assert torch.allclose(res, a + b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m, k, n", [(2, 2, 2), (10, 20, 5), (1, 10, 1)])
def test_matmul_shapes(m, k, n):
    # (M, K) x (K, N) -> (M, N)
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    res = npu.matmul(a, b)
    assert torch.allclose(res, torch.matmul(a, b), atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "s1, s2",
    [
        ((2, 3), (3,)),  # Add vector to matrix
        ((2, 3, 4), (1, 1, 4)),
        ((2, 3, 4), (1,)),
    ],
)
def test_broadcasting(s1, s2):
    # NPU Lib handles broadcasting via OpenVINO Add/Mul
    a = torch.randn(s1)
    b = torch.randn(s2)
    res = npu.add(a, b)
    assert torch.allclose(res, a + b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "s1, s2",
    [
        ((2, 3), (3,)),  # Add vector to matrix
        ((2, 3, 4), (1, 1, 4)),
        ((2, 3, 4), (1,)),
    ],
)
def test_broadcasting_backward(s1, s2):
    a1 = torch.randn(s1, requires_grad=True)
    b1 = torch.randn(s2, requires_grad=True)
    res1 = npu.add(a1, b1)
    res1.sum().backward()

    a2 = a1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)
    res2 = a2 + b2
    res2.sum().backward()

    assert torch.allclose(a1.grad, a2.grad, atol=1e-3)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-3)

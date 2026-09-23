import torch

import intel_npu_acceleration as npu


class TestNewOps:
    def test_squeeze(self):
        a = torch.randn(1, 10, 1, 20)
        # Squeeze all
        res = npu.squeeze(a)
        assert res.shape == (10, 20)
        assert torch.allclose(res, a.squeeze(), atol=1e-3)

        # Squeeze specific dim
        res = npu.squeeze(a, dim=0)
        assert res.shape == (10, 1, 20)
        assert torch.allclose(res, a.squeeze(0), atol=1e-3)

    def test_unsqueeze(self):
        a = torch.randn(10, 20)
        res = npu.unsqueeze(a, dim=1)
        assert res.shape == (10, 1, 20)
        assert torch.allclose(res, a.unsqueeze(1), atol=1e-3)

    def test_index_select(self):
        a = torch.randn(10, 20)
        indices = torch.tensor([1, 3, 5], dtype=torch.long)
        res = npu.index_select(a, 0, indices)
        assert res.shape == (3, 20)
        assert torch.allclose(res, torch.index_select(a, 0, indices), atol=1e-3)

    def test_zeros_ones_full(self):
        size = [2, 3]
        res = npu.zeros(size)
        assert res.shape == tuple(size)
        assert torch.all(res == 0)

        res = npu.ones(size)
        assert res.shape == tuple(size)
        assert torch.all(res == 1)

        res = npu.full(size, 3.14)
        assert res.shape == tuple(size)
        assert torch.allclose(res, torch.full(size, 3.14), atol=1e-3)

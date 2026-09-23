import pytest
import torch
import torch.nn as nn

from intel_npu_acceleration import compile_to_npu
from intel_npu_acceleration.functional import update_kv_cache


class FunctionalKVUpdate(nn.Module):
    def forward(self, cache, update, start_pos):
        # cache: (B, Max, H, D)
        # update: (B, Seq, H, D)
        return update_kv_cache(cache, update, start_pos)


class TestKVCache:
    def test_kv_update(self):
        model = FunctionalKVUpdate()
        model.eval()

        B, Max, H, D = 1, 10, 2, 4
        cache = torch.randn(B, Max, H, D)

        # Update at pos 2 with length 3
        start_pos = 2
        seq_len = 3
        update = torch.randn(B, seq_len, H, D)

        try:
            # Input signature: cache, update, start_pos
            # start_pos is int in python but we pass it as argument.
            # The compiler handles int args if they are part of example input.
            # FX traces 'start_pos' as input node.
            npu_model = compile_to_npu(model, (cache, update, start_pos))
        except Exception as e:
            pytest.fail(f"Compilation failed: {e}")

        out_npu = npu_model(cache, update, start_pos)
        out_cpu = model(cache, update, start_pos)

        assert torch.allclose(out_npu, out_cpu, atol=1e-3, rtol=1e-3)

        # Verify logic
        expected = cache.clone()
        expected[:, start_pos : start_pos + seq_len] = update
        assert torch.allclose(out_npu, expected, atol=1e-3, rtol=1e-3)

    def test_kv_update_eager(self):
        # Test direct eager execution with the functional interface
        B, Max, H, D = 2, 128, 8, 64
        cache = torch.randn(B, Max, H, D, dtype=torch.float32)

        # 1. Test basic eager update
        start_pos = 16
        seq_len = 8
        update = torch.randn(B, seq_len, H, D, dtype=torch.float32)

        out_eager = update_kv_cache(cache, update, start_pos)

        expected = cache.clone()
        expected[:, start_pos : start_pos + seq_len] = update
        assert torch.allclose(out_eager, expected, atol=1e-3, rtol=1e-3)

        # 2. Test eager update with float16
        cache_f16 = cache.half()
        update_f16 = update.half()
        out_eager_f16 = update_kv_cache(cache_f16, update_f16, start_pos)
        assert out_eager_f16.dtype == torch.float16
        assert torch.allclose(out_eager_f16.float(), expected, atol=1e-2, rtol=1e-2)

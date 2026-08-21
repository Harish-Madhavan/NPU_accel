"""
Stateful Key-Value Cache Module for Intel NPU LLM Acceleration.
"""

import torch
import torch.nn as nn


class NPUStatefulKVCache(nn.Module):
    """
    Stateful NPU Key-Value Cache Module.
    Keeps LLM KV cache data directly inside NPU hardware memory across decoding steps,
    eliminating expensive host-to-device context transfers.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # CPU eager fallback storage
        self.register_buffer(
            "cache",
            torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype),
        )
        self.register_buffer("pos", torch.zeros(1, dtype=torch.int32))
        self.current_pos = 0

    def forward(self, new_kv: torch.Tensor) -> torch.Tensor:
        seq_len = new_kv.shape[1]
        import intel_npu_acceleration.functional as npu_func

        if getattr(npu_func, "_IS_COMPILING", False) or torch.jit.is_tracing():
            pos_val = self.pos.long().squeeze()
            indices = torch.arange(0, seq_len, device=new_kv.device) + pos_val
            cache_clone = self.cache.clone()
            updated = cache_clone.index_copy(1, indices, new_kv)
            return updated
        else:
            # Fallback eager path (CPU execution / calibration)
            self.cache[:, self.current_pos : self.current_pos + seq_len] = new_kv
            self.current_pos += seq_len
            self.pos.fill_(int(self.current_pos))
            return self.cache

    def reset(self) -> None:
        """Reset the internal KV-cache state to zeros and index pointer to 0."""
        self.cache.zero_()
        self.pos.zero_()
        self.current_pos = 0

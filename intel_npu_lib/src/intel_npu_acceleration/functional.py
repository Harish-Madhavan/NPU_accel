import torch.fx
from . import _functional as F_base
from . import autograd as F_auto

# --- Combined API (Autograd-aware) ---

# Ops with full autograd (forward on NPU, backward computed on CPU)
add = F_auto.add
sub = F_auto.sub
mul = F_auto.mul
matmul = F_auto.matmul
linear = F_auto.linear
relu = F_auto.relu
gelu = F_auto.gelu  # was F_base — now gradient-aware
silu = F_auto.silu  # was F_base — now gradient-aware
rmsnorm = F_auto.rmsnorm  # was F_base — now gradient-aware
softmax = F_auto.softmax
conv2d = F_auto.conv2d  # was F_base — now gradient-aware (groups OK)
layer_norm = F_auto.layer_norm
hardsigmoid = F_auto.hardsigmoid
hardswish = F_auto.hardswish

# Ops without explicit autograd support (NPU dispatch, no gradient tape)
div = F_base.div
neg = F_base.neg
transpose = F_base.transpose
reshape = F_base.reshape
cat = F_base.cat
stack = F_base.stack
mean = F_base.mean
max_pool2d = F_base.max_pool2d
update_kv_cache = F_base.update_kv_cache
quantized_linear = F_base.quantized_linear
identity = F_base.identity
dropout = F_base.dropout
scaled_dot_product_attention = F_base.scaled_dot_product_attention

torch.fx.wrap(quantized_linear)

class NPUStatefulKVCache(torch.nn.Module):
    """
    Stateful NPU Key-Value Cache Module.
    Keeps LLM KV cache data directly inside NPU hardware memory across decoding steps,
    eliminating expensive host-to-device context transfers.
    """
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # CPU eager fallback storage
        self.register_buffer(
            "cache",
            torch.zeros(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype)
        )
        self.current_pos = 0

    def forward(self, new_kv: torch.Tensor) -> torch.Tensor:
        # Fallback eager path (CPU execution / calibration)
        seq_len = new_kv.shape[1]
        self.cache[:, self.current_pos : self.current_pos + seq_len] = new_kv
        self.current_pos += seq_len
        return self.cache

    def reset(self):
        """Reset the internal KV-cache state to zeros and index pointer to 0."""
        self.cache.zero_()
        self.current_pos = 0

"""
Intel NPU Functional Interface.

This module provides the primary functional API surface for Intel NPU accelerated operations.
Operations automatically dispatch to Level Zero hardware execution for both forward inference
and backward autograd passes when gradients are required.
"""

import torch.fx
from . import _functional as F_base
from . import autograd as F_auto

# --- Autograd-Aware Functional Operations (Forward and Backward accelerated on NPU) ---
add = F_auto.add
sub = F_auto.sub
mul = F_auto.mul
div = F_auto.div
neg = F_auto.neg
matmul = F_auto.matmul
linear = F_auto.linear
relu = F_auto.relu
gelu = F_auto.gelu
silu = F_auto.silu
softmax = F_auto.softmax
rmsnorm = F_auto.rmsnorm
layer_norm = F_auto.layer_norm
hardsigmoid = F_auto.hardsigmoid
hardswish = F_auto.hardswish
transpose = F_auto.transpose
reshape = F_auto.reshape
cat = F_auto.cat
stack = F_auto.stack
mean = F_auto.mean
embedding = F_auto.embedding
mse_loss = F_auto.mse_loss
conv2d = F_auto.conv2d
scaled_dot_product_attention = F_auto.scaled_dot_product_attention

# --- Structural & Eager NPU Operations ---
squeeze = F_base.squeeze
unsqueeze = F_base.unsqueeze
index_select = F_base.index_select
zeros = F_base.zeros
ones = F_base.ones
full = F_base.full
max_pool2d = F_base.max_pool2d
update_kv_cache = F_base.update_kv_cache
rotary_embedding = F_base.rotary_embedding
quantized_linear = F_base.quantized_linear
identity = F_base.identity
dropout = F_base.dropout

torch.fx.wrap(quantized_linear)
torch.fx.wrap(update_kv_cache)
torch.fx.wrap(rotary_embedding)

# Compilation flag to ensure functional execution during native OpenVINO tracing
_IS_COMPILING: bool = False

# Re-export NPUStatefulKVCache for backwards compatibility
from .nn.stateful_kv import NPUStatefulKVCache  # noqa: F401, E402

"""
Intel NPU Functional Interface.

This module provides the primary functional API surface for Intel NPU accelerated operations.
Operations automatically dispatch to Level Zero hardware execution for both forward inference
and backward autograd passes when gradients are required.
"""

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "matmul",
    "linear",
    "relu",
    "gelu",
    "silu",
    "softmax",
    "rmsnorm",
    "layer_norm",
    "hardsigmoid",
    "hardswish",
    "transpose",
    "reshape",
    "cat",
    "stack",
    "mean",
    "embedding",
    "mse_loss",
    "cross_entropy_loss",
    "l1_loss",
    "bce_with_logits_loss",
    "conv2d",
    "scaled_dot_product_attention",
    "squeeze",
    "unsqueeze",
    "index_select",
    "zeros",
    "ones",
    "full",
    "max_pool2d",
    "update_kv_cache",
    "rotary_embedding",
    "quantized_linear",
    "identity",
    "dropout",
    "NPUStatefulKVCache",
]

import contextlib
from collections.abc import Iterator

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
cross_entropy_loss = F_auto.cross_entropy_loss
l1_loss = F_auto.l1_loss
bce_with_logits_loss = F_auto.bce_with_logits_loss
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
# Use a simple bool for backward compat; compiler sets it via context manager
_IS_COMPILING: bool = False


@contextlib.contextmanager
def _compilation_context() -> Iterator[None]:
    """Context manager that sets _IS_COMPILING during OV conversion."""
    global _IS_COMPILING
    prev = _IS_COMPILING
    _IS_COMPILING = True
    try:
        yield
    finally:
        _IS_COMPILING = prev

# Re-export NPUStatefulKVCache for backwards compatibility
from .nn.stateful_kv import NPUStatefulKVCache  # noqa: F401, E402

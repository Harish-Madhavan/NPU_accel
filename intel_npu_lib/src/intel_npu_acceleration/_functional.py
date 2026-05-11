"""
_functional.py
--------------
Low-level functional interface that wraps the compiled C++ extension (_C).

All public functions in this module:
  - Accept PyTorch tensors of any supported dtype (float32, float16, int8/int32).
  - Normalise inputs to the dtype the C++ op currently expects.
  - Dispatch to _C.npu_*
  - Cast the result back to the original dtype when required.

This is an internal module. External code should use `functional.py` instead
(which adds autograd support on top of this layer).
"""

import torch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the compiled C++ extension
# ---------------------------------------------------------------------------
try:
    from . import _C
except ImportError as e:
    logger.error(
        f"Failed to import C++ extension '_C': {e}. "
        "Make sure the package was built (`python setup.py build_ext --inplace`)."
    )
    _C = None  # type: ignore[assignment]


def _require_C() -> None:
    if _C is None:
        raise RuntimeError(
            "Intel NPU C++ extension (_C) is not available. "
            "Re-build the extension with: python setup.py build_ext --inplace"
        )


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------


def _to_f32(t: torch.Tensor) -> torch.Tensor:
    """Cast tensor to float32 if it is not already float32."""
    return t if t.dtype == torch.float32 else t.float()


def _restore_dtype(result: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
    """Cast result back to the caller's original dtype."""
    if result.dtype == original_dtype:
        return result
    return result.to(original_dtype)


def _promote_binary(a: torch.Tensor, b: torch.Tensor):
    """
    Ensure both tensors share a common dtype that the C++ layer can handle.
    Returns (a_cast, b_cast, original_dtype)
    """
    if a.dtype == b.dtype:
        return a, b, a.dtype

    # Simple promotion: promote to the larger float type or float32 if mixed
    if a.is_floating_point() or b.is_floating_point():
        if a.dtype == torch.float64 or b.dtype == torch.float64:
            target = torch.float64
        elif a.dtype == torch.float32 or b.dtype == torch.float32:
            target = torch.float32
        elif a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16:
            target = torch.bfloat16
        else:
            target = torch.float16
    else:
        # Both integers
        target = torch.int32  # Default NPU int width
        if a.dtype == torch.int64 or b.dtype == torch.int64:
            target = torch.int64

    return a.to(target), b.to(target), target


# ---------------------------------------------------------------------------
# Element-wise binary ops
# ---------------------------------------------------------------------------


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.add(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_add(a, b), orig)


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.sub(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_sub(a, b), orig)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.mul(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_mul(a, b), orig)


def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.div(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_div(a, b), orig)


def neg(a: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.neg(a)
    return _C.npu_neg(a)


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.matmul(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_matmul(a, b), orig)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.linear(input, weight, bias)
    if bias is None:
        # C++ extension expects a bias tensor; pass an empty one if None
        bias = torch.empty(0, dtype=input.dtype)
    return _C.npu_linear(input, weight, bias)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def relu(a: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.relu(a)
    return _C.npu_relu(a)


def gelu(a: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.gelu(a)
    return _C.npu_gelu(a)


def silu(a: torch.Tensor) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.silu(a)
    return _C.npu_silu(a)


def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.softmax(a, dim=dim)
    return _C.npu_softmax(a, dim)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if _C is None:
        # Pure-PyTorch fallback
        rms = torch.sqrt(input.float().pow(2).mean(-1, keepdim=True) + eps)
        return ((input.float() / rms) * weight.float()).to(input.dtype)
    return _C.npu_rmsnorm(input, weight, eps)


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


def transpose(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    if _C is None:
        return torch.transpose(input, dim0, dim1)
    rank = input.dim()
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    perm = list(range(rank))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return _C.npu_transpose(input, perm)


def reshape(input: torch.Tensor, shape: List[int]) -> torch.Tensor:
    if _C is None:
        return torch.reshape(input, shape)
    return _C.npu_reshape(input, list(shape))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    if attn_mask is None:
        attn_mask = torch.empty(0, dtype=query.dtype)
    return _C.npu_scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


# ---------------------------------------------------------------------------
# KV-cache update (slice-assign pattern used in autoregressive LLM inference)
# ---------------------------------------------------------------------------



def update_kv_cache(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    position: int,
) -> torch.Tensor:
    seq_len = new_kv.shape[1]
    indices = torch.arange(position, position + seq_len, dtype=torch.long)
    return cache.index_copy(1, indices, new_kv)


# ---------------------------------------------------------------------------
# CV ops
# ---------------------------------------------------------------------------


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.conv2d(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _to_pair(v):
        return [v, v] if isinstance(v, int) else list(v)

    if bias is None:
        bias = torch.empty(0, dtype=input.dtype)

    return _C.npu_conv2d(
        input,
        weight,
        bias,
        _to_pair(stride),
        _to_pair(padding),
        _to_pair(dilation),
        groups,
    )


def max_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if _C is None:
        return torch.nn.functional.max_pool2d(
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    def _to_pair(v):
        return [v, v] if isinstance(v, int) else list(v)

    if stride is None:
        stride = kernel_size

    return _C.npu_max_pool2d(
        input,
        _to_pair(kernel_size),
        _to_pair(stride),
        _to_pair(padding),
        _to_pair(dilation),
        ceil_mode,
    )


# ---------------------------------------------------------------------------
# No-ops (pass-through) — present so the functional API is complete
# ---------------------------------------------------------------------------


def identity(input: torch.Tensor) -> torch.Tensor:
    return input


def dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    """During inference (training=False) dropout is a no-op."""
    if training:
        # Actual stochastic dropout is not offloaded to the NPU — use PyTorch.
        return torch.nn.functional.dropout(input, p=p, training=True, inplace=inplace)
    return input

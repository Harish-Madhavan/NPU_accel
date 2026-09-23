import logging
import sys
from typing import Any

import torch
import torch.fx

logger = logging.getLogger(__name__)

# Cache the torch.compiler.is_compiling callable once at import so every
# eager op doesn't pay hasattr + attribute-lookup overhead on the hot path.
_TORCH_IS_COMPILING: Any = getattr(getattr(torch, "compiler", None), "is_compiling", None)
_TORCH_JIT_IS_TRACING: Any = getattr(getattr(torch, "jit", None), "is_tracing", None)

try:
    from intel_npu_acceleration.device import _C
except Exception:
    try:
        from intel_npu_acceleration import _C
    except ImportError as e:
        logger.warning(f"C++ extension '_C' not available ({e}); falling back to OpenVINO CPU/NPU path.")
        _C = None
    except Exception as e:
        logger.warning(f"Failed to load C++ extension '_C': {e}")
        _C = None


def _to_f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def _restore_dtype(res: torch.Tensor, orig: torch.dtype) -> torch.Tensor:
    return res if res.dtype == orig else res.to(orig)


def _to_pair(v: Any) -> list[int]:
    return [v, v] if isinstance(v, int) else list(v)  # type: ignore[return-value]


def _should_bypass_c() -> bool:
    npu_func = sys.modules.get("intel_npu_acceleration.functional")
    if npu_func is not None and getattr(npu_func, "_IS_COMPILING", False):
        return True
    # Modern torch: use cached compiler is_compiling if available
    if _TORCH_IS_COMPILING is not None:
        try:
            if _TORCH_IS_COMPILING():
                return True
        except Exception:
            pass
    if _TORCH_JIT_IS_TRACING is not None:
        try:
            if _TORCH_JIT_IS_TRACING():  # legacy fallback
                return True
        except Exception:
            pass
    return False


def _is_proxy(*args) -> bool:
    if _should_bypass_c():
        return True
    for arg in args:
        if isinstance(arg, torch.fx.Proxy):
            return True
        if isinstance(arg, (list, tuple)) and _is_proxy(*arg):
            return True
    return False


def _promote_binary(a: torch.Tensor, b: torch.Tensor):
    """Ensure both tensors share a common dtype for C++ backend execution using PyTorch native promotion."""
    if a.dtype == b.dtype:
        return a, b, a.dtype
    target = torch.promote_types(a.dtype, b.dtype)
    return a.to(target), b.to(target), target


# Sentinel returned by _try_npu when the hardware path is unavailable.
_MISS: Any = object()

_REDUCTION_CODES = {"none": 0, "mean": 1, "sum": 2}


def _reduction_code(reduction: str) -> int:
    """Map a PyTorch reduction string to the integer code the C++ extension expects."""
    return _REDUCTION_CODES.get(reduction, 1)


def _try_npu(method: str, *args: Any) -> Any:
    """Call ``_C.<method>(*args)``; return ``_MISS`` unless the NPU path succeeds.

    Centralizes the dispatch guard repeated by every op: skip when the
    extension is missing (or lacks the method), when tracing/proxy tensors
    are present, or when the native call raises — the caller then runs its
    pure-PyTorch fallback.
    """
    if _C is None or _is_proxy(*args):
        return _MISS
    fn = getattr(_C, method, None)
    if fn is None:
        return _MISS
    try:
        return fn(*args)
    except Exception:
        return _MISS


def _or_empty(t: torch.Tensor | None, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    """Return ``t``, or an empty placeholder tensor the C++ extension treats as "absent".

    The native kernels take tensors (not ``None``) for optional bias /
    mask / scale arguments; an empty tensor signals "not provided".
    """
    if t is not None:
        return t
    if device is None:
        return torch.empty(0, dtype=dtype)
    return torch.empty(0, dtype=dtype, device=device)


def unpack_int4_packed(weight_u8: torch.Tensor) -> torch.Tensor:
    """Unpack nibble-interleaved uint8 weights to a float row-major matrix.

    Each byte packs two 4-bit values (low nibble = even element, high nibble
    = odd element). Shared by the eager quantized-linear fallback and the
    graph-mode dequantization pass so both paths unpack identically.
    """
    # Cast to int32 first: the NPU Floor op has no unsigned element types.
    w_s32 = weight_u8.to(torch.int32)
    w_odd = torch.floor_divide(w_s32, 16)
    w_even = w_s32 - w_odd * 16
    return torch.stack([w_even, w_odd], dim=-1).view(weight_u8.shape[0], -1).float()

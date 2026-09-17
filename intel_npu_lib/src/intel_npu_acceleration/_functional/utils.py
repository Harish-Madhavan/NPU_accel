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

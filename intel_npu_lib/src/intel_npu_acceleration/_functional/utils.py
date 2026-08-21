import sys
import torch
import torch.fx
import logging

logger = logging.getLogger(__name__)

try:
    from .. import _C
except ImportError as e:
    logger.error(f"Failed to import C++ extension '_C': {e}.")
    _C = None


def _to_f32(t: torch.Tensor) -> torch.Tensor: return t if t.dtype == torch.float32 else t.float()
def _restore_dtype(res: torch.Tensor, orig: torch.dtype) -> torch.Tensor: return res if res.dtype == orig else res.to(orig)
def _to_pair(v): return [v, v] if isinstance(v, int) else list(v)


def _should_bypass_C() -> bool:
    npu_func = sys.modules.get("intel_npu_acceleration.functional")
    if npu_func is not None and getattr(npu_func, "_IS_COMPILING", False):
        return True
    return torch.jit.is_tracing()


def _is_proxy(*args) -> bool:
    if _should_bypass_C():
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

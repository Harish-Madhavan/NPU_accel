import torch
import torch.fx
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the compiled C++ extension
# ---------------------------------------------------------------------------
try:
    from .. import _C
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


def _to_f32(t: torch.Tensor) -> torch.Tensor:
    """Cast tensor to float32 if it is not already float32."""
    return t if t.dtype == torch.float32 else t.float()


def _restore_dtype(result: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
    """Cast result back to the caller's original dtype."""
    if result.dtype == original_dtype:
        return result
    return result.to(original_dtype)


def _is_proxy(*args) -> bool:
    for arg in args:
        if isinstance(arg, torch.fx.Proxy):
            return True
        if isinstance(arg, (list, tuple)):
            if _is_proxy(*arg):
                return True
    return False


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

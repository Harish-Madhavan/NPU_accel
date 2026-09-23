import torch

from .utils import _C, _is_proxy


def _unary_op(a: torch.Tensor, torch_fn, npu_fn) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch_fn(a)
    return npu_fn(a)

def relu(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.relu, _C.npu_relu if _C else None)


def gelu(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.nn.functional.gelu, _C.npu_gelu if _C else None)


def silu(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.nn.functional.silu, _C.npu_silu if _C else None)


def hardsigmoid(a: torch.Tensor) -> torch.Tensor:
    if _C is not None and hasattr(_C, "npu_hardsigmoid") and not _is_proxy(a):
        return _C.npu_hardsigmoid(a)
    return torch.nn.functional.hardsigmoid(a)


def hardswish(a: torch.Tensor) -> torch.Tensor:
    if _C is not None and hasattr(_C, "npu_hardswish") and not _is_proxy(a):
        return _C.npu_hardswish(a)
    return torch.nn.functional.hardswish(a)


def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.nn.functional.softmax(a, dim=dim)
    return _C.npu_softmax(a, dim)


def neg(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.neg, _C.npu_neg if _C else None)


def sin(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.sin, _C.npu_sin if _C else None)


def cos(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.cos, _C.npu_cos if _C else None)


def exp(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.exp, _C.npu_exp if _C else None)


def sqrt(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.sqrt, _C.npu_sqrt if _C else None)


def abs(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.abs, _C.npu_abs if _C else None)


def rsqrt(a: torch.Tensor) -> torch.Tensor:
    return _unary_op(a, torch.rsqrt, _C.npu_rsqrt if _C else None)


def clamp(a: torch.Tensor, min_val: float | None = None, max_val: float | None = None) -> torch.Tensor:
    if min_val is None and max_val is None:
        raise ValueError("clamp: at least one of 'min_val' or 'max_val' must not be None")
    lo = float("-inf") if min_val is None else float(min_val)
    hi = float("inf") if max_val is None else float(max_val)
    if _C is None or _is_proxy(a):
        return torch.clamp(a, lo, hi)
    return _C.npu_clamp(a, lo, hi)

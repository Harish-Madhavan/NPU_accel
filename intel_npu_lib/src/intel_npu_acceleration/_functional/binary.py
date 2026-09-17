import torch

from .utils import _C, _is_proxy, _promote_binary, _restore_dtype


def _binary_op(a: torch.Tensor, b: torch.Tensor, torch_fn, npu_fn) -> torch.Tensor:
    if _C is None or _is_proxy(a, b) or a.numel() == 0 or b.numel() == 0:
        return torch_fn(a, b)
    a_cast, b_cast, orig_dtype = _promote_binary(a, b)
    return _restore_dtype(npu_fn(a_cast, b_cast), orig_dtype)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, torch.add, _C.npu_add if _C else None)


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, torch.sub, _C.npu_sub if _C else None)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, torch.mul, _C.npu_mul if _C else None)


def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, torch.div, _C.npu_div if _C else None)

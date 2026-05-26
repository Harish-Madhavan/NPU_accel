import torch
from .utils import _C, _is_proxy, _promote_binary, _restore_dtype


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b) or a.numel() == 0 or b.numel() == 0:
        return torch.add(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_add(a, b), orig)


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b) or a.numel() == 0 or b.numel() == 0:
        return torch.sub(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_sub(a, b), orig)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b) or a.numel() == 0 or b.numel() == 0:
        return torch.mul(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_mul(a, b), orig)


def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b) or a.numel() == 0 or b.numel() == 0:
        return torch.div(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_div(a, b), orig)

import torch
from .utils import _C, _is_proxy


def relu(a: torch.Tensor) -> torch.Tensor:
    if _C is None or isinstance(a, torch.fx.Proxy):
        return torch.relu(a)
    return _C.npu_relu(a)


def gelu(a: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.nn.functional.gelu(a)
    return _C.npu_gelu(a)


def silu(a: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.nn.functional.silu(a)
    return _C.npu_silu(a)


def hardsigmoid(a: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.nn.functional.hardsigmoid(a)
    return torch.nn.functional.hardsigmoid(a)


def hardswish(a: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.nn.functional.hardswish(a)
    return torch.nn.functional.hardswish(a)


def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _C is None or isinstance(a, torch.fx.Proxy):
        return torch.nn.functional.softmax(a, dim=dim)
    return _C.npu_softmax(a, dim)


def neg(a: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a):
        return torch.neg(a)
    return _C.npu_neg(a)

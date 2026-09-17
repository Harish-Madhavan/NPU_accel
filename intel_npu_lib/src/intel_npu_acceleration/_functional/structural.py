from typing import List, Optional

import torch

from .utils import _C, _is_proxy


def transpose(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    if _C is None or _is_proxy(input):
        return torch.transpose(input, dim0, dim1)
    rank = input.dim()
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    perm = list(range(rank))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return _C.npu_transpose(input, perm)


def reshape(input: torch.Tensor, shape: list[int]) -> torch.Tensor:
    if _C is None or _is_proxy(input) or input.numel() == 0:
        return torch.reshape(input, shape)
    return _C.npu_reshape(input, list(shape))


def squeeze(input: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if _C is None or _is_proxy(input) or input.numel() == 0:
        return torch.squeeze(input, dim) if dim is not None else torch.squeeze(input)
    dims = [dim] if dim is not None else []
    return _C.npu_squeeze(input, dims)


def unsqueeze(input: torch.Tensor, dim: int) -> torch.Tensor:
    if _C is None or _is_proxy(input) or input.numel() == 0:
        return torch.unsqueeze(input, dim)
    return _C.npu_unsqueeze(input, [dim])


def cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if dim < 0 and len(tensors) > 0:
        dim = dim + len(tensors[0].shape)
    if (
        _C is None
        or _is_proxy(*tensors)
        or any(t.numel() == 0 for t in tensors)
    ):
        return torch.cat(tensors, dim=dim)
    return _C.npu_cat(tensors, dim)


def stack(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if dim < 0 and len(tensors) > 0:
        dim = dim + len(tensors[0].shape) + 1
    if (
        _C is None
        or _is_proxy(*tensors)
        or any(t.numel() == 0 for t in tensors)
    ):
        return torch.stack(tensors, dim=dim)
    return _C.npu_stack(tensors, dim)


def mean(
    input: torch.Tensor, dim: list[int] | None = None, keepdim: bool = False
) -> torch.Tensor:
    if _C is None or _is_proxy(input) or input.numel() == 0:
        if dim is None:
            return torch.mean(input)
        return torch.mean(input, dim=dim, keepdim=keepdim)
    if dim is None:
        dim_list = []
    elif isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    return _C.npu_mean(input, dim_list, keepdim)


def index_select(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(input, index) or input.numel() == 0 or index.numel() == 0:
        return torch.index_select(input, dim, index)
    return _C.npu_index_select(input, dim, index)


def zeros(size: list[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype)


def ones(size: list[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones(size, dtype=dtype)


def full(size: list[int], fill_value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.full(size, fill_value, dtype=dtype)


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

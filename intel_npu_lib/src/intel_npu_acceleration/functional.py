import torch
from typing import Optional, List, Union, Tuple
import warnings

try:
    from . import _C
except ImportError as e:
    warnings.warn(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
    _C = None

def _safe_call(name: str, *args) -> Optional[torch.Tensor]:
    if _C and hasattr(_C, name):
        return getattr(_C, name)(*args)
    return None

def _to_list(val: Union[int, Tuple[int, ...], List[int]], n: int = 2) -> List[int]:
    if isinstance(val, int):
        return [val] * n
    return list(val)

# --- Eager Execution Wrappers ---

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_add', a, b)
    return res if res is not None else a + b

def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_sub', a, b)
    return res if res is not None else a - b

def neg(a: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_neg', a)
    return res if res is not None else torch.neg(a)

def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_mul', a, b)
    return res if res is not None else a * b

def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_div', a, b)
    return res if res is not None else a / b

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_matmul', a, b)
    return res if res is not None else torch.matmul(a, b)

def relu(a: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_relu', a)
    return res if res is not None else torch.relu(a)

def gelu(a: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_gelu', a)
    return res if res is not None else torch.nn.functional.gelu(a)

def silu(a: torch.Tensor) -> torch.Tensor:
    res = _safe_call('npu_silu', a)
    return res if res is not None else torch.nn.functional.silu(a)

def softmax(a: torch.Tensor, dim: int) -> torch.Tensor:
    res = _safe_call('npu_softmax', a, dim)
    return res if res is not None else torch.softmax(a, dim)

def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if bias is None:
        bias = torch.zeros(weight.size(0), dtype=input.dtype, device=input.device)
    res = _safe_call('npu_linear', input, weight, bias)
    return res if res is not None else torch.nn.functional.linear(input, weight, bias)

def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    res = _safe_call('npu_rmsnorm', input, weight, eps)
    if res is not None:
        return res
    variance = input.pow(2).mean(-1, keepdim=True)
    input_norm = input * torch.rsqrt(variance + eps)
    return input_norm * weight

def transpose(a: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    rank = a.dim()
    perm = list(range(rank))
    if dim0 < 0: dim0 += rank
    if dim1 < 0: dim1 += rank
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    res = _safe_call('npu_transpose', a, perm)
    return res if res is not None else torch.transpose(a, dim0, dim1)

def reshape(a: torch.Tensor, shape: Union[List[int], Tuple[int, ...], torch.Size]) -> torch.Tensor:
    if isinstance(shape, (list, tuple, torch.Size)):
        target_shape = list(shape)
    else:
        target_shape = list(shape)
    res = _safe_call('npu_reshape', a, target_shape)
    return res if res is not None else a.reshape(target_shape)

def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
) -> torch.Tensor:
    
    stride_list = _to_list(stride)
    padding_list = _to_list(padding)
    dilation_list = _to_list(dilation)
    
    if bias is None:
        bias = torch.Tensor().to(input.device) 
    
    res = _safe_call(
        'npu_conv2d', 
        input, weight, bias, 
        stride_list, padding_list, dilation_list, groups
    )
    return res if res is not None else torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)

def max_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> torch.Tensor:
    
    if return_indices:
        return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

    k_list = _to_list(kernel_size)
    s_list = _to_list(stride if stride is not None else kernel_size)
    p_list = _to_list(padding)
    d_list = _to_list(dilation)
    
    res = _safe_call(
        'npu_max_pool2d',
        input, k_list, s_list, p_list, d_list, ceil_mode
    )
    return res if res is not None else torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

# --- Utilities ---

def update_kv_cache(cache: torch.Tensor, update: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
    """
    Update the Key/Value cache at specific positions using functional ops (Cat/Slice).
    """
    prefix = cache[:, :start_pos] 
    suffix = cache[:, start_pos + seqlen:]
    return torch.cat([prefix, update, suffix], dim=1)

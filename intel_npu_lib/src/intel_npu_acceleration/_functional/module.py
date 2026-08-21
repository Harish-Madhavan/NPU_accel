import warnings
import torch
from typing import Optional, List
from .utils import _C, _is_proxy, _promote_binary, _restore_dtype, _to_pair


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b):
        return torch.matmul(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_matmul(a, b), orig)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (
        _C is None
        or _is_proxy(input, weight, bias)
    ):
        return torch.nn.functional.linear(input, weight, bias)
    if bias is None:
        # C++ extension expects a bias tensor; pass an empty one if None
        bias = torch.empty(0, dtype=input.dtype)
    return _C.npu_linear(input, weight, bias)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if _C is None or _is_proxy(input, weight):
        # Pure-PyTorch fallback
        rms = torch.sqrt(input.float().pow(2).mean(-1, keepdim=True) + eps)
        return ((input.float() / rms) * weight.float()).to(input.dtype)
    return _C.npu_rmsnorm(input, weight, eps)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if _C is None or _is_proxy(input, weight, bias):
        return torch.nn.functional.layer_norm(
            input, normalized_shape, weight, bias, eps
        )
    if weight is None:
        weight = torch.empty(0, dtype=input.dtype)
    if bias is None:
        bias = torch.empty(0, dtype=input.dtype)
    return _C.npu_layer_norm(input, list(normalized_shape), weight, bias, eps)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
) -> torch.Tensor:
    if _C is None or _is_proxy(query, key, value, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    if attn_mask is None:
        attn_mask = torch.empty(0, dtype=query.dtype)
    return _C.npu_scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )


def update_kv_cache(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    position: int,
) -> torch.Tensor:
    if (
        _C is None
        or _is_proxy(cache, new_kv, position)
    ):
        seq_len = new_kv.shape[1]
        if isinstance(position, (int, float)):
            indices = torch.arange(position, position + seq_len, dtype=torch.long)
        elif isinstance(position, torch.Tensor) and not isinstance(
            position, torch.fx.Proxy
        ) and not torch.jit.is_tracing():
            indices = torch.arange(
                position.item(), position.item() + seq_len, dtype=torch.long
            )
        else:
            indices = (
                torch.arange(seq_len, dtype=torch.long, device=new_kv.device) + position
            )
        return cache.index_copy(1, indices, new_kv)

    if isinstance(position, int):
        pos_tensor = torch.tensor(position, dtype=torch.long, device=cache.device)
    else:
        if isinstance(position, torch.Tensor) and position.dim() > 0:
            pos_tensor = position.squeeze()
        else:
            pos_tensor = position

    return _C.npu_update_kv_cache(cache, new_kv, pos_tensor)


def quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if any(isinstance(arg, torch.fx.Proxy) for arg in (input, weight, scale, zero_point, bias)):
        out_features = weight.shape[0]
        out_shape = input.shape[:-1] + (out_features,)
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

        is_int4 = (
            weight.dtype == torch.uint8
            and len(weight.shape) == 2
            and weight.shape[1] == input.shape[-1] // 2
        )

        if _C is None or _is_proxy(input, weight, scale, zero_point, bias):
            if is_int4:
                # Cast weight to int32 to satisfy NPU Floor op element type requirements (no unsigned types)
                weight_s32 = weight.to(torch.int32)
                w_odd = torch.floor_divide(weight_s32, 16)
                w_even = weight_s32 - w_odd * 16
                w_even_u = w_even.unsqueeze(1)
                w_odd_u = w_odd.unsqueeze(1)
                w_unpacked = torch.cat([w_even_u, w_odd_u], dim=1).transpose(1, 2).reshape(weight.shape[0], -1)
                w_float = w_unpacked.float()
            else:
                w_float = weight.float()

            if zero_point is not None and zero_point.numel() > 0:
                w_float = w_float - zero_point.float()
            w_float = w_float * scale.float()
            w_float = w_float.to(input.dtype)
            return torch.nn.functional.linear(input, w_float, bias)

    if zero_point is None:
        zero_point = torch.empty(0, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = torch.empty(0, dtype=input.dtype, device=input.device)

    return _C.npu_quantized_linear(input, weight, scale, zero_point, bias)


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> torch.Tensor:
    if _C is None or isinstance(input, torch.fx.Proxy):
        return torch.nn.functional.conv2d(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    if bias is None:
        bias = torch.empty(0, dtype=input.dtype)

    return _C.npu_conv2d(
        input,
        weight,
        bias,
        _to_pair(stride),
        _to_pair(padding),
        _to_pair(dilation),
        groups,
    )


def max_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if _C is None or isinstance(input, torch.fx.Proxy):
        return torch.nn.functional.max_pool2d(
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    if stride is None:
        stride = kernel_size

    return _C.npu_max_pool2d(
        input,
        _to_pair(kernel_size),
        _to_pair(stride),
        _to_pair(padding),
        _to_pair(dilation),
        ceil_mode,
    )

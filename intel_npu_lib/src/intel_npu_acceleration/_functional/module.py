import warnings
from typing import List, Optional

import torch

from .utils import _C, _is_proxy, _promote_binary, _restore_dtype, _to_pair


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(a, b):
        return torch.matmul(a, b)
    a, b, orig = _promote_binary(a, b)
    return _restore_dtype(_C.npu_matmul(a, b), orig)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
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
    normalized_shape: list[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
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
    attn_mask: torch.Tensor | None = None,
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
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
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


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Hardware accelerated Rotary Position Embedding (RoPE) for LLMs."""
    if _C is not None and not _is_proxy(x, cos, sin):
        try:
            return _C.npu_rotary_embedding(x, cos, sin)
        except Exception:
            pass
    # PyTorch CPU reference fallback
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
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

    return _C.npu_max_pool2d(
        input,
        _to_pair(kernel_size),
        _to_pair(stride),
        _to_pair(padding),
        _to_pair(dilation),
        ceil_mode,
    )


def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if _C is None or _is_proxy(weight, indices):
        return torch.nn.functional.embedding(indices, weight)
    return _C.npu_embedding(weight, indices)


def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if _C is None or _is_proxy(pred, target):
        return torch.nn.functional.mse_loss(pred, target, reduction=reduction)
    red_map = {"none": 0, "mean": 1, "sum": 2}
    return _C.npu_mse_loss(pred, target, red_map.get(reduction, 1))


def cross_entropy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if _C is None or _is_proxy(pred, target):
        return torch.nn.functional.cross_entropy(pred, target, reduction=reduction)
    red_map = {"none": 0, "mean": 1, "sum": 2}
    try:
        return _C.npu_cross_entropy_loss(pred, target, red_map.get(reduction, 1))
    except Exception:
        return torch.nn.functional.cross_entropy(pred, target, reduction=reduction)


def cross_entropy_loss_backward(
    grad_output: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if _C is not None and not _is_proxy(grad_output, pred, target):
        red_map = {"none": 0, "mean": 1, "sum": 2}
        try:
            return _C.npu_cross_entropy_loss_backward(grad_output, pred, target, red_map.get(reduction, 1))
        except Exception:
            pass
    sm = torch.softmax(pred, dim=-1)
    if target.dim() == pred.dim():
        target_prob = target
    else:
        target_prob = torch.zeros_like(pred).scatter_(-1, target.unsqueeze(-1), 1.0)
    grad = sm - target_prob
    if reduction == "mean":
        batch_size = pred.shape[0] if pred.dim() > 1 else 1
        grad = grad / batch_size
    return grad * grad_output


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute L1 loss (Mean Absolute Error) accelerated on Intel NPU."""
    if _C is None or _is_proxy(pred, target):
        return torch.nn.functional.l1_loss(pred, target, reduction=reduction)
    red_map = {"none": 0, "mean": 1, "sum": 2}
    try:
        return _C.npu_l1_loss(pred, target, red_map.get(reduction, 1))
    except Exception:
        return torch.nn.functional.l1_loss(pred, target, reduction=reduction)


def l1_loss_backward(
    grad_output: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute backward gradient for L1 loss."""
    if _C is not None and not _is_proxy(grad_output, pred, target):
        red_map = {"none": 0, "mean": 1, "sum": 2}
        try:
            return _C.npu_l1_loss_backward(grad_output, pred, target, red_map.get(reduction, 1))
        except Exception:
            pass
    diff = pred - target
    sgn = torch.sign(diff)
    if reduction == "mean":
        sgn = sgn / pred.numel()
    return sgn * grad_output


def bce_with_logits_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Binary Cross Entropy with Logits Loss."""
    return torch.nn.functional.binary_cross_entropy_with_logits(
        input, target, weight=weight, reduction=reduction, pos_weight=pos_weight
    )


def bce_with_logits_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute backward gradient for Binary Cross Entropy with Logits Loss."""
    sig = torch.sigmoid(input)
    if pos_weight is not None:
        grad = sig * (1.0 + (pos_weight - 1.0) * target) - target * pos_weight
    else:
        grad = sig - target
    if weight is not None:
        grad = grad * weight
    if reduction == "mean":
        grad = grad / input.numel()
    return grad * grad_output



def matmul_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor]:
    if _C is not None and not _is_proxy(grad_output, a, b):
        try:
            return _C.npu_matmul_backward(grad_output, a, b)
        except Exception:
            pass
    grad_a = grad_output @ b.transpose(-2, -1)
    grad_b = a.transpose(-2, -1) @ grad_output
    return [grad_a, grad_b]


def linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
    needs_bias_grad: bool = True,
) -> list[torch.Tensor]:
    if _C is not None and not _is_proxy(grad_output, input, weight):
        try:
            return _C.npu_linear_backward(
                grad_output, input, weight, needs_input_grad, needs_weight_grad, needs_bias_grad
            )
        except Exception:
            pass
    grad_input = grad_output @ weight if needs_input_grad else torch.empty(0)
    grad_weight = grad_output.reshape(-1, grad_output.size(-1)).t() @ input.reshape(-1, input.size(-1)) if needs_weight_grad else torch.empty(0)
    grad_bias = grad_output.reshape(-1, grad_output.size(-1)).sum(0) if needs_bias_grad else torch.empty(0)
    return [grad_input, grad_weight, grad_bias]


def relu_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    if _C is not None and not _is_proxy(grad_output, input):
        try:
            return _C.npu_relu_backward(grad_output, input)
        except Exception:
            pass
    grad = grad_output.clone()
    grad[input <= 0] = 0
    return grad


def gelu_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    if _C is not None and not _is_proxy(grad_output, input):
        try:
            return _C.npu_gelu_backward(grad_output, input)
        except Exception:
            pass
    # Exact GELU derivative fallback
    import math
    c1 = 1.0 / math.sqrt(2.0)
    c2 = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(c1 * input))
    pdf = c2 * torch.exp(-0.5 * input.pow(2))
    return grad_output * (cdf + input * pdf)


def silu_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    if _C is not None and not _is_proxy(grad_output, input):
        try:
            return _C.npu_silu_backward(grad_output, input)
        except Exception:
            pass
    s = torch.sigmoid(input)
    return grad_output * (s * (1.0 + input * (1.0 - s)))


def softmax_backward(grad_output: torch.Tensor, output: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _C is not None and not _is_proxy(grad_output, output):
        try:
            return _C.npu_softmax_backward(grad_output, output, dim)
        except Exception:
            pass
    sum_gy = (grad_output * output).sum(dim=dim, keepdim=True)
    return output * (grad_output - sum_gy)


def rmsnorm_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> list[torch.Tensor]:
    if _C is not None and not _is_proxy(grad_output, input, weight):
        try:
            return _C.npu_rmsnorm_backward(grad_output, input, weight, eps)
        except Exception:
            pass
    go = grad_output.float()
    in_f = input.float()
    w_f = weight.float()
    rms = torch.sqrt(in_f.pow(2).mean(-1, keepdim=True) + eps)
    xn = in_f / rms
    grad_w = (go * xn).sum(dim=tuple(range(go.dim() - 1))).to(weight.dtype)
    dl_dxn = go * w_f
    correction = (dl_dxn * xn).mean(-1, keepdim=True)
    grad_in = ((dl_dxn - xn * correction) / rms).to(grad_output.dtype)
    return [grad_in, grad_w]


def adam_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> list[torch.Tensor]:
    if _C is not None and not _is_proxy(param, grad, exp_avg, exp_avg_sq):
        try:
            return _C.npu_adam_step(
                param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step
            )
        except Exception:
            pass
    if weight_decay != 0:
        grad = grad + param * weight_decay
    exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + (grad * grad) * (1.0 - beta2)
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    step_val = (exp_avg / bc1) / (torch.sqrt(exp_avg_sq / bc2) + eps)
    param = param - step_val * lr
    return [param, exp_avg, exp_avg_sq]


def sgd_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: float,
    nesterov: bool,
    has_momentum_buffer: bool,
) -> list[torch.Tensor]:
    if _C is not None and not _is_proxy(param, grad, momentum_buffer):
        try:
            return _C.npu_sgd_step(
                param,
                grad,
                momentum_buffer,
                lr,
                momentum,
                weight_decay,
                dampening,
                nesterov,
                has_momentum_buffer,
            )
        except Exception:
            pass
    effective_grad = grad
    if weight_decay != 0.0:
        effective_grad = grad + param * weight_decay
    new_buf = momentum_buffer
    if momentum != 0.0:
        if not has_momentum_buffer:
            new_buf = effective_grad.clone()
        else:
            new_buf = momentum_buffer * momentum + effective_grad * (1.0 - dampening)
        if nesterov:
            update_dir = effective_grad + new_buf * momentum
        else:
            update_dir = new_buf
    else:
        update_dir = effective_grad
    param = param - update_dir * lr
    return [param, new_buf]



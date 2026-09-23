import torch
import torch.fx

from intel_npu_acceleration import _functional as F_npu

from .functions import (
    NPUAbs,
    NPUAdd,
    NPUBCEWithLogitsLoss,
    NPUCat,
    NPUClamp,
    NPUConv2d,
    NPUCos,
    NPUCrossEntropyLoss,
    NPUDiv,
    NPUEmbedding,
    NPUExp,
    NPUGeLU,
    NPUHardSigmoid,
    NPUHardSwish,
    NPUL1Loss,
    NPULayerNorm,
    NPULinear,
    NPUMatMul,
    NPUMaxPool2d,
    NPUMean,
    NPUMSELoss,
    NPUMul,
    NPUNeg,
    NPUPow,
    NPUReLU,
    NPUReshape,
    NPURMSNorm,
    NPURsqrt,
    NPUScaledDotProductAttention,
    NPUSiLU,
    NPUSin,
    NPUSoftmax,
    NPUSqrt,
    NPUStack,
    NPUSub,
    NPUTranspose,
    NPUTriu,
    NPUWhere,
)

__all__ = [
    "matmul",
    "add",
    "sub",
    "mul",
    "relu",
    "gelu",
    "silu",
    "sin",
    "cos",
    "exp",
    "sqrt",
    "abs",
    "rsqrt",
    "pow",
    "clamp",
    "where",
    "triu",
    "flatten",
    "rmsnorm",
    "softmax",
    "max_pool2d",
    "linear",
    "conv2d",
    "layer_norm",
    "hardsigmoid",
    "hardswish",
    "neg",
    "div",
    "transpose",
    "reshape",
    "cat",
    "stack",
    "mean",
    "embedding",
    "mse_loss",
    "cross_entropy_loss",
    "l1_loss",
    "bce_with_logits_loss",
    "scaled_dot_product_attention",
]


def _should_use_autograd(*args) -> bool:
    """Helper to check if autograd dispatch is needed and no FX Proxy objects are present."""
    has_grad = False
    for a in args:
        if isinstance(a, torch.fx.Proxy):
            return False
        if isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, torch.fx.Proxy):
                    return False
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    has_grad = True
        elif isinstance(a, torch.Tensor) and a.requires_grad:
            has_grad = True
    return has_grad


def matmul(a, b):
    return NPUMatMul.apply(a, b) if _should_use_autograd(a, b) else F_npu.matmul(a, b)


def add(a, b):
    return NPUAdd.apply(a, b) if _should_use_autograd(a, b) else F_npu.add(a, b)


def sub(a, b):
    return NPUSub.apply(a, b) if _should_use_autograd(a, b) else F_npu.sub(a, b)


def mul(a, b):
    return NPUMul.apply(a, b) if _should_use_autograd(a, b) else F_npu.mul(a, b)


def relu(a):
    return NPUReLU.apply(a) if _should_use_autograd(a) else F_npu.relu(a)


def gelu(a):
    return NPUGeLU.apply(a) if _should_use_autograd(a) else F_npu.gelu(a)


def silu(a):
    return NPUSiLU.apply(a) if _should_use_autograd(a) else F_npu.silu(a)


def sin(a):
    return NPUSin.apply(a) if _should_use_autograd(a) else F_npu.sin(a)


def cos(a):
    return NPUCos.apply(a) if _should_use_autograd(a) else F_npu.cos(a)


def exp(a):
    return NPUExp.apply(a) if _should_use_autograd(a) else F_npu.exp(a)


def sqrt(a):
    return NPUSqrt.apply(a) if _should_use_autograd(a) else F_npu.sqrt(a)


def abs(a):
    return NPUAbs.apply(a) if _should_use_autograd(a) else F_npu.abs(a)


def rsqrt(a):
    return NPURsqrt.apply(a) if _should_use_autograd(a) else F_npu.rsqrt(a)


def pow(a, exponent):
    return NPUPow.apply(a, exponent) if _should_use_autograd(a, exponent) else F_npu.pow(a, exponent)


def clamp(a, min=None, max=None):
    if min is None and max is None:
        raise ValueError("clamp: at least one of 'min' or 'max' must not be None")
    return NPUClamp.apply(a, min, max) if _should_use_autograd(a) else F_npu.clamp(a, min, max)


def where(condition, a, b):
    return (
        NPUWhere.apply(condition, a, b)
        if _should_use_autograd(condition, a, b)
        else F_npu.where(condition, a, b)
    )


def triu(input, diagonal=0):
    return NPUTriu.apply(input, diagonal) if _should_use_autograd(input) else F_npu.triu(input, diagonal)


def flatten(input, start_dim=0, end_dim=-1):
    shape = list(input.shape)
    rank = len(shape)
    s = start_dim + rank if start_dim < 0 else start_dim
    e = end_dim + rank if end_dim < 0 else end_dim
    flat = 1
    for d in shape[s : e + 1]:
        flat *= d
    new_shape = shape[:s] + [flat] + shape[e + 1 :]
    if _should_use_autograd(input):
        return NPUReshape.apply(input, new_shape)
    return F_npu.reshape(input, new_shape)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    if _should_use_autograd(input):
        return NPUMaxPool2d.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    return F_npu.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


def rmsnorm(input, weight, eps=1e-6):
    return NPURMSNorm.apply(input, weight, eps) if _should_use_autograd(input, weight) else F_npu.rmsnorm(input, weight, eps)


def softmax(a, dim=-1):
    return NPUSoftmax.apply(a, dim) if _should_use_autograd(a) else F_npu.softmax(a, dim)


def linear(input, weight, bias=None):
    return NPULinear.apply(input, weight, bias) if _should_use_autograd(input, weight, bias) else F_npu.linear(input, weight, bias)


def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    return NPUConv2d.apply(input, weight, bias, stride, padding, dilation, groups) if _should_use_autograd(input, weight, bias) else F_npu.conv2d(input, weight, bias, stride, padding, dilation, groups)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return NPULayerNorm.apply(input, normalized_shape, weight, bias, eps) if _should_use_autograd(input, weight, bias) else F_npu.layer_norm(input, normalized_shape, weight, bias, eps)


def hardsigmoid(a):
    return NPUHardSigmoid.apply(a) if _should_use_autograd(a) else F_npu.hardsigmoid(a)


def hardswish(a):
    return NPUHardSwish.apply(a) if _should_use_autograd(a) else F_npu.hardswish(a)


def neg(a):
    return NPUNeg.apply(a) if _should_use_autograd(a) else F_npu.neg(a)


def div(a, b):
    return NPUDiv.apply(a, b) if _should_use_autograd(a, b) else F_npu.div(a, b)


def transpose(input, dim0, dim1):
    return NPUTranspose.apply(input, dim0, dim1) if _should_use_autograd(input) else F_npu.transpose(input, dim0, dim1)


def reshape(input, shape):
    return NPUReshape.apply(input, shape) if _should_use_autograd(input) else F_npu.reshape(input, shape)


def cat(tensors, dim=0):
    return NPUCat.apply(dim, *tensors) if _should_use_autograd(tensors) else F_npu.cat(tensors, dim)


def stack(tensors, dim=0):
    return NPUStack.apply(dim, *tensors) if _should_use_autograd(tensors) else F_npu.stack(tensors, dim)


def mean(input, dim=None, keepdim=False):
    return NPUMean.apply(input, dim, keepdim) if _should_use_autograd(input) else F_npu.mean(input, dim, keepdim)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    if _should_use_autograd(input, weight):
        return NPUEmbedding.apply(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return torch.nn.functional.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)


def mse_loss(pred, target, reduction="mean"):
    if _should_use_autograd(pred, target):
        return NPUMSELoss.apply(pred, target, reduction)
    return F_npu.mse_loss(pred, target, reduction=reduction)


def cross_entropy_loss(pred, target, reduction="mean"):
    if _should_use_autograd(pred, target):
        return NPUCrossEntropyLoss.apply(pred, target, reduction)
    return F_npu.cross_entropy_loss(pred, target, reduction=reduction)


def l1_loss(pred, target, reduction="mean"):
    if _should_use_autograd(pred, target):
        return NPUL1Loss.apply(pred, target, reduction)
    return F_npu.l1_loss(pred, target, reduction=reduction)


def bce_with_logits_loss(input, target, weight=None, reduction="mean", pos_weight=None):
    if _should_use_autograd(input, target):
        return NPUBCEWithLogitsLoss.apply(input, target, weight, reduction, pos_weight)
    return F_npu.bce_with_logits_loss(input, target, weight=weight, reduction=reduction, pos_weight=pos_weight)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if _should_use_autograd(query, key, value):
        return NPUScaledDotProductAttention.apply(query, key, value, attn_mask, dropout_p, is_causal, scale)
    return F_npu.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale if scale is not None else 0.0
    )


# Wrap public functions to prevent FX tracing into requires_grad checks
for _fn in (
    matmul,
    add,
    sub,
    mul,
    relu,
    gelu,
    silu,
    sin,
    cos,
    exp,
    sqrt,
    abs,
    rsqrt,
    pow,
    clamp,
    where,
    triu,
    flatten,
    rmsnorm,
    softmax,
    max_pool2d,
    linear,
    conv2d,
    layer_norm,
    hardsigmoid,
    hardswish,
    neg,
    div,
    transpose,
    reshape,
    cat,
    stack,
    mean,
    embedding,
    mse_loss,
    cross_entropy_loss,
    l1_loss,
    bce_with_logits_loss,
    scaled_dot_product_attention,
):
    torch.fx.wrap(_fn)
del _fn


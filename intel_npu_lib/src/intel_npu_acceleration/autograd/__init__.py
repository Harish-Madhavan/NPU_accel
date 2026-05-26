import torch
import torch.fx
from .. import _functional as F_npu
from .functions import (
    NPUMatMul,
    NPUAdd,
    NPUSub,
    NPUMul,
    NPUReLU,
    NPUSoftmax,
    NPULinear,
    NPUGeLU,
    NPUSiLU,
    NPURMSNorm,
    NPUConv2d,
    NPULayerNorm,
    NPUHardSigmoid,
    NPUHardSwish,
    NPUNeg,
    NPUDiv,
    NPUTranspose,
    NPUReshape,
    NPUCat,
    NPUStack,
    NPUMean,
    NPUEmbedding,
    NPUScaledDotProductAttention,
)


def matmul(a, b):
    if not isinstance(a, torch.fx.Proxy) and (a.requires_grad or b.requires_grad):
        return NPUMatMul.apply(a, b)
    return F_npu.matmul(a, b)


def add(a, b):
    if not isinstance(a, torch.fx.Proxy) and (a.requires_grad or b.requires_grad):
        return NPUAdd.apply(a, b)
    return F_npu.add(a, b)


def sub(a, b):
    if not isinstance(a, torch.fx.Proxy) and (a.requires_grad or b.requires_grad):
        return NPUSub.apply(a, b)
    return F_npu.sub(a, b)


def mul(a, b):
    if not isinstance(a, torch.fx.Proxy) and (a.requires_grad or b.requires_grad):
        return NPUMul.apply(a, b)
    return F_npu.mul(a, b)


def relu(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUReLU.apply(a)
    return F_npu.relu(a)


def gelu(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUGeLU.apply(a)
    return F_npu.gelu(a)


def silu(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUSiLU.apply(a)
    return F_npu.silu(a)


def rmsnorm(input, weight, eps=1e-6):
    if not isinstance(input, torch.fx.Proxy) and (
        input.requires_grad or weight.requires_grad
    ):
        return NPURMSNorm.apply(input, weight, eps)
    return F_npu.rmsnorm(input, weight, eps)


def softmax(a, dim=-1):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUSoftmax.apply(a, dim)
    return F_npu.softmax(a, dim)


def linear(input, weight, bias=None):
    if not isinstance(input, torch.fx.Proxy) and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return NPULinear.apply(input, weight, bias)
    return F_npu.linear(input, weight, bias)


def conv2d(
    input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    if not isinstance(input, torch.fx.Proxy) and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return NPUConv2d.apply(input, weight, bias, stride, padding, dilation, groups)
    return F_npu.conv2d(input, weight, bias, stride, padding, dilation, groups)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if not isinstance(input, torch.fx.Proxy) and (
        input.requires_grad
        or (weight is not None and weight.requires_grad)
        or (bias is not None and bias.requires_grad)
    ):
        return NPULayerNorm.apply(input, normalized_shape, weight, bias, eps)
    return F_npu.layer_norm(input, normalized_shape, weight, bias, eps)


def hardsigmoid(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUHardSigmoid.apply(a)
    return F_npu.hardsigmoid(a)


def hardswish(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUHardSwish.apply(a)
    return F_npu.hardswish(a)


def neg(a):
    if not isinstance(a, torch.fx.Proxy) and a.requires_grad:
        return NPUNeg.apply(a)
    return F_npu.neg(a)


def div(a, b):
    if not isinstance(a, torch.fx.Proxy) and (
        (isinstance(a, torch.Tensor) and a.requires_grad)
        or (isinstance(b, torch.Tensor) and b.requires_grad)
    ):
        return NPUDiv.apply(a, b)
    return F_npu.div(a, b)


def transpose(input, dim0, dim1):
    if not isinstance(input, torch.fx.Proxy) and input.requires_grad:
        return NPUTranspose.apply(input, dim0, dim1)
    return F_npu.transpose(input, dim0, dim1)


def reshape(input, shape):
    if not isinstance(input, torch.fx.Proxy) and input.requires_grad:
        return NPUReshape.apply(input, shape)
    return F_npu.reshape(input, shape)


def cat(tensors, dim=0):
    any_requires_grad = False
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            any_requires_grad = True
            break

    if not isinstance(tensors, torch.fx.Proxy) and any_requires_grad:
        has_proxy = False
        for t in tensors:
            if isinstance(t, torch.fx.Proxy):
                has_proxy = True
                break
        if not has_proxy:
            return NPUCat.apply(dim, *tensors)
    return F_npu.cat(tensors, dim)


def stack(tensors, dim=0):
    any_requires_grad = False
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            any_requires_grad = True
            break

    if not isinstance(tensors, torch.fx.Proxy) and any_requires_grad:
        has_proxy = False
        for t in tensors:
            if isinstance(t, torch.fx.Proxy):
                has_proxy = True
                break
        if not has_proxy:
            return NPUStack.apply(dim, *tensors)
    return F_npu.stack(tensors, dim)


def mean(input, dim=None, keepdim=False):
    if not isinstance(input, torch.fx.Proxy) and input.requires_grad:
        return NPUMean.apply(input, dim, keepdim)
    return F_npu.mean(input, dim, keepdim)


def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    if not isinstance(input, torch.fx.Proxy) and (
        (isinstance(input, torch.Tensor) and input.requires_grad)
        or (isinstance(weight, torch.Tensor) and weight.requires_grad)
    ):
        return NPUEmbedding.apply(
            input,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        )
    return torch.nn.functional.embedding(
        input,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    if not isinstance(query, torch.fx.Proxy) and (
        query.requires_grad
        or key.requires_grad
        or value.requires_grad
    ):
        return NPUScaledDotProductAttention.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    return F_npu.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale if scale is not None else 0.0,
    )


# Wrap public functions to prevent FX tracing into requires_grad checks
torch.fx.wrap(matmul)
torch.fx.wrap(add)
torch.fx.wrap(sub)
torch.fx.wrap(mul)
torch.fx.wrap(relu)
torch.fx.wrap(gelu)
torch.fx.wrap(silu)
torch.fx.wrap(rmsnorm)
torch.fx.wrap(softmax)
torch.fx.wrap(linear)
torch.fx.wrap(conv2d)
torch.fx.wrap(layer_norm)
torch.fx.wrap(hardsigmoid)
torch.fx.wrap(hardswish)
torch.fx.wrap(neg)
torch.fx.wrap(div)
torch.fx.wrap(transpose)
torch.fx.wrap(reshape)
torch.fx.wrap(cat)
torch.fx.wrap(stack)
torch.fx.wrap(mean)
torch.fx.wrap(embedding)
torch.fx.wrap(scaled_dot_product_attention)

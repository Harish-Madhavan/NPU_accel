import torch
import torch.fx

from intel_npu_acceleration import _functional as F_npu

from .functions import (
    NPUAdd,
    NPUBCEWithLogitsLoss,
    NPUCat,
    NPUConv2d,
    NPUCrossEntropyLoss,
    NPUDiv,
    NPUEmbedding,
    NPUGeLU,
    NPUHardSigmoid,
    NPUHardSwish,
    NPUL1Loss,
    NPULayerNorm,
    NPULinear,
    NPUMatMul,
    NPUMean,
    NPUMSELoss,
    NPUMul,
    NPUNeg,
    NPUReLU,
    NPUReshape,
    NPURMSNorm,
    NPUScaledDotProductAttention,
    NPUSiLU,
    NPUSoftmax,
    NPUStack,
    NPUSub,
    NPUTranspose,
)


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
torch.fx.wrap(mse_loss)
torch.fx.wrap(cross_entropy_loss)
torch.fx.wrap(l1_loss)
torch.fx.wrap(bce_with_logits_loss)
torch.fx.wrap(scaled_dot_product_attention)


import torch
import math
from . import _functional as F_npu


def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    grad_dim = len(grad.shape)
    target_dim = len(target_shape)
    if grad_dim > target_dim:
        grad = grad.sum(dim=tuple(range(grad_dim - target_dim)))
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


class NPUMatMul(torch.autograd.Function):
    """
    NPU-accelerated Matrix Multiplication with Autograd support.
    Forward pass runs on NPU; backward pass computes gradients on CPU.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.transpose(-2, -1)
        grad_b = a.transpose(-2, -1) @ grad_output
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class NPUAdd(torch.autograd.Function):
    """NPU-accelerated Addition with Autograd support."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.add(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return _unbroadcast(grad_output, a.shape), _unbroadcast(grad_output, b.shape)


class NPUSub(torch.autograd.Function):
    """NPU-accelerated Subtraction with Autograd support."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.sub(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return _unbroadcast(grad_output, a.shape), _unbroadcast(-grad_output, b.shape)


class NPUMul(torch.autograd.Function):
    """NPU-accelerated Multiplication with Autograd support."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.mul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class NPUReLU(torch.autograd.Function):
    """NPU-accelerated ReLU with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F_npu.relu(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = grad_output.clone()
        grad_a[a < 0] = 0
        return grad_a


class NPUSoftmax(torch.autograd.Function):
    """NPU-accelerated Softmax with Autograd support."""

    @staticmethod
    def forward(ctx, a, dim):
        out = F_npu.softmax(a, dim)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        # softmax gradient: s * (g - (g * s).sum(dim, keepdim=True))
        sum_grad_out_s = (grad_output * out).sum(ctx.dim, keepdim=True)
        grad_a = out * (grad_output - sum_grad_out_s)
        return grad_a, None


class NPULinear(torch.autograd.Function):
    """
    NPU-accelerated Linear layer (y = xW^T + b) with Autograd support.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        return F_npu.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.transpose(-2, -1) @ input
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# GeLU  —  d/dx GeLU(x) = 0.5*(1 + erf(x/√2)) + x·exp(-x²/2)/√(2π)
# ---------------------------------------------------------------------------


class NPUGeLU(torch.autograd.Function):
    """NPU-accelerated GeLU with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F_npu.gelu(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a32 = a.float()
        cdf = 0.5 * (1.0 + torch.erf(a32 / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * a32 * a32) / math.sqrt(2.0 * math.pi)
        grad_a = (cdf + a32 * pdf).to(grad_output.dtype) * grad_output
        return grad_a


# ---------------------------------------------------------------------------
# SiLU (Swish)  —  d/dx SiLU(x) = σ(x)·(1 + x·(1 − σ(x)))
# ---------------------------------------------------------------------------


class NPUSiLU(torch.autograd.Function):
    """NPU-accelerated SiLU (Swish) with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F_npu.silu(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        sig = torch.sigmoid(a.float())
        grad_a = (sig * (1.0 + a.float() * (1.0 - sig))).to(
            grad_output.dtype
        ) * grad_output
        return grad_a


# ---------------------------------------------------------------------------
# RMSNorm  —  y = (x / rms(x)) * w,   rms(x) = sqrt(mean(x²) + ε)
# ---------------------------------------------------------------------------


class NPURMSNorm(torch.autograd.Function):
    """NPU-accelerated RMSNorm with Autograd support."""

    @staticmethod
    def forward(ctx, input, weight, eps):
        out = F_npu.rmsnorm(input, weight, eps)
        rms = torch.sqrt(input.float().pow(2).mean(-1, keepdim=True) + eps)
        x_norm = (input.float() / rms).to(input.dtype)
        ctx.save_for_backward(x_norm, weight)
        ctx.rms = rms
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, weight = ctx.saved_tensors
        rms = ctx.rms

        go = grad_output.float()
        w = weight.float()
        xn = x_norm.float()

        grad_weight = (go * xn).sum(dim=tuple(range(go.dim() - 1))).to(weight.dtype)

        dL_dxn = go * w
        correction = (dL_dxn * xn).mean(dim=-1, keepdim=True)
        grad_input = ((dL_dxn - xn * correction) / rms).to(grad_output.dtype)

        return grad_input, grad_weight, None


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------


class NPUConv2d(torch.autograd.Function):
    """NPU-accelerated Conv2d with Autograd support."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return F_npu.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape,
                weight,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input,
                weight.shape,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None


# ---------------------------------------------------------------------------
# Public autograd-aware wrappers
# Routes to the Function class only when gradients are actually required,
# so there is zero autograd overhead during pure inference.
# ---------------------------------------------------------------------------


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
    if not isinstance(input, torch.fx.Proxy) and (input.requires_grad or weight.requires_grad):
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


import torch.fx

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

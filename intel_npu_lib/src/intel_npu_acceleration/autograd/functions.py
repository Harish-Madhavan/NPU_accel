import math

import torch

from intel_npu_acceleration import _functional as F_npu


def _unbroadcast(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
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
    Both forward and backward passes execute on the Intel NPU.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a, grad_b = F_npu.matmul_backward(grad_output, a, b)
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


def _make_unary_function(name: str, forward_fn, backward_fn) -> type:
    """Build a single-input/single-output autograd Function class.

    NPUReLU / NPUGeLU / NPUSiLU share identical forward/backward structure
    and differ only in the dispatched ``F_npu`` kernels; generating them
    keeps the three definitions from drifting apart.
    """

    class NPUUnary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            ctx.save_for_backward(a)
            return forward_fn(a)

        @staticmethod
        def backward(ctx, grad_output):
            (a,) = ctx.saved_tensors
            return backward_fn(grad_output, a)

    NPUUnary.__name__ = name
    NPUUnary.__qualname__ = name
    return NPUUnary


NPUReLU = _make_unary_function("NPUReLU", F_npu.relu, F_npu.relu_backward)
"""NPU-accelerated ReLU with Autograd support."""

NPUGeLU = _make_unary_function("NPUGeLU", F_npu.gelu, F_npu.gelu_backward)
"""NPU-accelerated GeLU with Autograd support."""

NPUSiLU = _make_unary_function("NPUSiLU", F_npu.silu, F_npu.silu_backward)
"""NPU-accelerated SiLU (Swish) with Autograd support."""

NPUSin = _make_unary_function("NPUSin", F_npu.sin, F_npu.sin_backward)
"""NPU-accelerated Sine with Autograd support."""

NPUCos = _make_unary_function("NPUCos", F_npu.cos, F_npu.cos_backward)
"""NPU-accelerated Cosine with Autograd support."""

NPUExp = _make_unary_function("NPUExp", F_npu.exp, F_npu.exp_backward)
"""NPU-accelerated Exponential with Autograd support."""

NPUSqrt = _make_unary_function("NPUSqrt", F_npu.sqrt, F_npu.sqrt_backward)
"""NPU-accelerated Square Root with Autograd support."""

NPUAbs = _make_unary_function("NPUAbs", F_npu.abs, F_npu.abs_backward)
"""NPU-accelerated Absolute Value with Autograd support."""

NPURsqrt = _make_unary_function("NPURsqrt", F_npu.rsqrt, F_npu.rsqrt_backward)
"""NPU-accelerated Reciprocal Square Root with Autograd support."""


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
        grad_a = F_npu.softmax_backward(grad_output, out, ctx.dim)
        return grad_a, None


class NPULinear(torch.autograd.Function):
    """
    NPU-accelerated Linear layer (y = xW^T + b) with Level Zero Autograd support.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        return F_npu.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        needs_input = ctx.needs_input_grad[0]
        needs_weight = ctx.needs_input_grad[1]
        needs_bias = ctx.needs_input_grad[2] if bias is not None else False
        grad_input, grad_weight, grad_bias = F_npu.linear_backward(
            grad_output, input, weight, needs_input, needs_weight, needs_bias
        )
        return (
            grad_input if needs_input else None,
            grad_weight if needs_weight else None,
            grad_bias if (bias is not None and needs_bias) else None,
        )


class NPURMSNorm(torch.autograd.Function):
    """NPU-accelerated RMSNorm with Autograd support."""

    @staticmethod
    def forward(ctx, input, weight, eps):
        out = F_npu.rmsnorm(input, weight, eps)
        ctx.save_for_backward(input, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_in, grad_w = F_npu.rmsnorm_backward(grad_output, input, weight, ctx.eps)
        return grad_in, grad_w, None


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
        grad_input, grad_weight, grad_bias = F_npu.conv2d_backward(
            grad_output,
            input,
            weight,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
            needs_input_grad=ctx.needs_input_grad[0],
            needs_weight_grad=ctx.needs_input_grad[1],
            needs_bias_grad=bias is not None and ctx.needs_input_grad[2],
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class NPULayerNorm(torch.autograd.Function):
    """NPU-accelerated LayerNorm with Autograd support."""

    @staticmethod
    def forward(ctx, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        out = F_npu.layer_norm(input, normalized_shape, weight, bias, eps)

        rank = input.dim()
        k = len(normalized_shape)
        axes = tuple(range(rank - k, rank))

        mean = input.float().mean(dim=axes, keepdim=True)
        var = input.float().var(dim=axes, keepdim=True, unbiased=False)
        rms = torch.sqrt(var + eps)
        x_norm = ((input.float() - mean) / rms).to(input.dtype)

        ctx.save_for_backward(x_norm, weight, bias)
        ctx.rms = rms
        ctx.axes = axes
        ctx.normalized_shape = normalized_shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_norm, weight, bias = ctx.saved_tensors
        rms = ctx.rms
        axes = ctx.axes
        normalized_shape = ctx.normalized_shape

        go = grad_output.float()
        xn = x_norm.float()

        non_norm_dims = tuple(range(go.dim() - len(normalized_shape)))

        grad_weight = None
        if weight is not None and ctx.needs_input_grad[2]:
            grad_weight = (go * xn).sum(dim=non_norm_dims).to(weight.dtype)

        grad_bias = None
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = go.sum(dim=non_norm_dims).to(bias.dtype)

        grad_input = None
        if ctx.needs_input_grad[0]:
            w = weight.float() if weight is not None else 1.0
            dl_dxn = go * w

            n_elements = math.prod(normalized_shape)
            mean_dl_dxn = dl_dxn.sum(dim=axes, keepdim=True) / n_elements
            mean_dl_dxn_xn = (dl_dxn * xn).sum(dim=axes, keepdim=True) / n_elements

            grad_input = ((dl_dxn - mean_dl_dxn - xn * mean_dl_dxn_xn) / rms).to(grad_output.dtype)

        return grad_input, None, grad_weight, grad_bias, None


class NPUHardSigmoid(torch.autograd.Function):
    """NPU-accelerated HardSigmoid with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F_npu.hardsigmoid(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = torch.where((-3.0 < a) & (a < 3.0), grad_output / 6.0, 0.0)
        return grad_a


class NPUHardSwish(torch.autograd.Function):
    """NPU-accelerated HardSwish with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F_npu.hardswish(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        h = torch.clamp(a + 3.0, 0.0, 6.0) / 6.0
        slope = torch.where((-3.0 < a) & (a < 3.0), a / 6.0, 0.0)
        grad_a = (h + slope) * grad_output
        return grad_a


class NPUNeg(torch.autograd.Function):
    """NPU-accelerated Negation with Autograd support."""

    @staticmethod
    def forward(ctx, a):
        return F_npu.neg(a)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class NPUClamp(torch.autograd.Function):
    """NPU-accelerated Clamp with Autograd support."""

    @staticmethod
    def forward(ctx, a, min_val, max_val):
        ctx.save_for_backward(a)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return F_npu.clamp(a, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        lo = float("-inf") if ctx.min_val is None else float(ctx.min_val)
        hi = float("inf") if ctx.max_val is None else float(ctx.max_val)
        return F_npu.clamp_backward(grad_output, a, lo, hi), None, None


class NPUPow(torch.autograd.Function):
    """NPU-accelerated Power with Autograd support."""

    @staticmethod
    def forward(ctx, a, exponent):
        ctx.exponent_is_tensor = isinstance(exponent, torch.Tensor)
        if ctx.exponent_is_tensor:
            ctx.save_for_backward(a, exponent)
        else:
            ctx.save_for_backward(a)
            ctx.exponent = exponent
        return F_npu.pow(a, exponent)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        a = saved[0]
        if ctx.exponent_is_tensor:
            exponent = saved[1]
            needs_b = ctx.needs_input_grad[1]
        else:
            exponent = torch.full((), ctx.exponent, dtype=a.dtype, device=a.device)
            needs_b = False
        grad_a, grad_b = F_npu.pow_backward(
            grad_output,
            a,
            exponent,
            needs_a_grad=ctx.needs_input_grad[0],
            needs_exponent_grad=needs_b,
        )
        return grad_a, grad_b


class NPUWhere(torch.autograd.Function):
    """NPU-accelerated Where/Select with Autograd support."""

    @staticmethod
    def forward(ctx, condition, a, b):
        ctx.save_for_backward(condition)
        return F_npu.where(condition, a, b)

    @staticmethod
    def backward(ctx, grad_output):
        (condition,) = ctx.saved_tensors
        grad_a, grad_b = F_npu.where_backward(grad_output, condition)
        return None, grad_a, grad_b


class NPUTriu(torch.autograd.Function):
    """NPU-accelerated Upper-Triangular with Autograd support."""

    @staticmethod
    def forward(ctx, a, diagonal=0):
        ctx.diagonal = diagonal
        return F_npu.triu(a, diagonal)

    @staticmethod
    def backward(ctx, grad_output):
        return F_npu.triu_backward(grad_output, ctx.diagonal), None


class NPUMaxPool2d(torch.autograd.Function):
    """NPU-accelerated MaxPool2d with Autograd support."""

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        return F_npu.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = F_npu.max_pool2d_backward(
            grad_output, input, ctx.kernel_size, ctx.stride, ctx.padding,
            ctx.dilation, ctx.ceil_mode,
        )
        return grad_input, None, None, None, None, None


class NPUDiv(torch.autograd.Function):
    """NPU-accelerated Division with Autograd support."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F_npu.div(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output / b
        grad_b = -grad_output * a / (b * b)
        return _unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape)


class NPUTranspose(torch.autograd.Function):
    """NPU-accelerated Transpose with Autograd support."""

    @staticmethod
    def forward(ctx, input, dim0, dim1):
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return F_npu.transpose(input, dim0, dim1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.transpose(ctx.dim0, ctx.dim1), None, None


class NPUReshape(torch.autograd.Function):
    """NPU-accelerated Reshape with Autograd support."""

    @staticmethod
    def forward(ctx, input, shape):
        ctx.input_shape = input.shape
        return F_npu.reshape(input, shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.reshape(ctx.input_shape), None


class NPUCat(torch.autograd.Function):
    """NPU-accelerated Concatenate with Autograd support."""

    @staticmethod
    def forward(ctx, dim, *tensors):
        ctx.dim = dim
        ctx.tensor_sizes = [t.shape[dim] for t in tensors]
        return F_npu.cat(list(tensors), dim)

    @staticmethod
    def backward(ctx, grad_output):
        grads = torch.split(grad_output, ctx.tensor_sizes, dim=ctx.dim)
        return (None,) + tuple(grads)


class NPUStack(torch.autograd.Function):
    """NPU-accelerated Stack with Autograd support."""

    @staticmethod
    def forward(ctx, dim, *tensors):
        ctx.dim = dim
        return F_npu.stack(list(tensors), dim)

    @staticmethod
    def backward(ctx, grad_output):
        grads = torch.unbind(grad_output, dim=ctx.dim)
        return (None,) + tuple(grads)


class NPUMean(torch.autograd.Function):
    """NPU-accelerated Mean with Autograd support."""

    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        return F_npu.mean(input, dim, keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim

        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
            numel = math.prod(input_shape)
            grad_input = grad_output.expand(input_shape) / numel
        else:
            if isinstance(dim, int):
                dims = [dim]
            else:
                dims = list(dim)

            rank = len(input_shape)
            dims = [d + rank if d < 0 else d for d in dims]

            reduced_elements = 1
            for d in dims:
                reduced_elements *= input_shape[d]

            if not keepdim:
                grad_reshape = list(grad_output.shape)
                for d in sorted(dims):
                    grad_reshape.insert(d, 1)
                grad_output = grad_output.reshape(grad_reshape)

            grad_input = grad_output.expand(input_shape) / reduced_elements

        return grad_input, None, None


class NPUEmbedding(torch.autograd.Function):
    """NPU-accelerated Embedding with Autograd support."""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        ctx.save_for_backward(input, weight)
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return torch.nn.functional.embedding(
            input,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        num_embeddings = weight.shape[0]
        grad_weight = F_npu.embedding_backward(
            grad_output,
            input,
            num_embeddings,
            padding_idx=ctx.padding_idx,
            scale_grad_by_freq=ctx.scale_grad_by_freq,
            sparse=ctx.sparse,
        )
        return None, grad_weight, None, None, None, None, None


class NPUScaledDotProductAttention(torch.autograd.Function):
    """NPU-accelerated Scaled Dot Product Attention with Autograd support."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale

        out = F_npu.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale if scale is not None else 0.0,
        )

        ctx.save_for_backward(query, key, value, attn_mask, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, attn_mask, out = ctx.saved_tensors
        grad_query, grad_key, grad_value = F_npu.scaled_dot_product_attention_backward(
            grad_output,
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale if ctx.scale is not None else 0.0,
        )
        return grad_query, grad_key, grad_value, None, None, None, None


class NPUMSELoss(torch.autograd.Function):
    """NPU-accelerated MSE loss with Autograd support."""

    @staticmethod
    def forward(ctx, pred, target, reduction="mean"):
        ctx.save_for_backward(pred, target)
        ctx.reduction = reduction
        return F_npu.mse_loss(pred, target, reduction=reduction)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        grad_pred = F_npu.mse_loss_backward(grad_output, pred, target, reduction=ctx.reduction)
        grad_target = -grad_pred
        return grad_pred, grad_target, None


class NPUCrossEntropyLoss(torch.autograd.Function):
    """NPU-accelerated Cross Entropy loss with Autograd support."""

    @staticmethod
    def forward(ctx, pred, target, reduction="mean"):
        ctx.save_for_backward(pred, target)
        ctx.reduction = reduction
        return F_npu.cross_entropy_loss(pred, target, reduction=reduction)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        red = ctx.reduction
        grad_pred = F_npu.cross_entropy_loss_backward(grad_output, pred, target, reduction=red)
        return grad_pred, None, None


class NPUL1Loss(torch.autograd.Function):
    """NPU-accelerated L1 loss with Autograd support."""

    @staticmethod
    def forward(ctx, pred, target, reduction="mean"):
        ctx.save_for_backward(pred, target)
        ctx.reduction = reduction
        return F_npu.l1_loss(pred, target, reduction=reduction)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        red = ctx.reduction
        grad_pred = F_npu.l1_loss_backward(grad_output, pred, target, reduction=red)
        grad_target = -grad_pred
        return grad_pred, grad_target, None


class NPUBCEWithLogitsLoss(torch.autograd.Function):
    """NPU-accelerated Binary Cross Entropy with Logits Loss with Autograd support."""

    @staticmethod
    def forward(ctx, input, target, weight=None, reduction="mean", pos_weight=None):
        ctx.save_for_backward(input, target, weight, pos_weight)
        ctx.reduction = reduction
        return F_npu.bce_with_logits_loss(
            input, target, weight=weight, reduction=reduction, pos_weight=pos_weight
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight, pos_weight = ctx.saved_tensors
        red = ctx.reduction
        grad_input = F_npu.bce_with_logits_loss_backward(
            grad_output, input, target, weight=weight, reduction=red, pos_weight=pos_weight
        )
        return grad_input, None, None, None, None




import torch
import math
from .. import _functional as F_npu


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
            dL_dxn = go * w

            N = math.prod(normalized_shape)
            mean_dL_dxn = dL_dxn.sum(dim=axes, keepdim=True) / N
            mean_dL_dxn_xn = (dL_dxn * xn).sum(dim=axes, keepdim=True) / N

            grad_input = ((dL_dxn - mean_dL_dxn - xn * mean_dL_dxn_xn) / rms).to(grad_output.dtype)

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
        padding_idx = ctx.padding_idx if ctx.padding_idx is not None else -1
        grad_weight = torch.ops.aten.embedding_backward(
            grad_output,
            input,
            num_embeddings,
            padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
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

        with torch.enable_grad():
            q = query.detach().requires_grad_(True)
            k = key.detach().requires_grad_(True)
            v = value.detach().requires_grad_(True)

            out_cpu = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=ctx.dropout_p,
                is_causal=ctx.is_causal,
                scale=ctx.scale,
            )

            out_cpu.backward(grad_output)

            grad_query = q.grad
            grad_key = k.grad
            grad_value = v.grad

        return grad_query, grad_key, grad_value, None, None, None, None

"""
Intel NPU PyTorch Neural Network Modules (Drop-in replacements with Level Zero hardware acceleration).

This module provides standard `torch.nn.Module` subclasses engineered for execution
on Intel NPUs via oneAPI Level Zero. These modules can be used as direct drop-in
replacements for standard PyTorch layers during training and inference.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple, Any
from .. import functional as F_npu


class Linear(nn.Module):
    """NPU-accelerated Linear transformation layer ($y = xA^T + b$).

    Applies a linear transformation to incoming data with both forward pass
    and backward autograd execution accelerated on the Intel NPU.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        device (optional): Target device for parameter allocation. Defaults to None.
        dtype (optional): Desired floating point dtype. Defaults to None.

    Attributes:
        weight (torch.nn.Parameter): The learnable weights of the module of shape `(out_features, in_features)`.
        bias (torch.nn.Parameter): The learnable bias of the module of shape `(out_features)`.

    Examples:
        >>> from intel_npu_acceleration.nn import Linear
        >>> layer = Linear(128, 64)
        >>> x = torch.randn(8, 128)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([8, 64])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize layer parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute linear projection on NPU."""
        return F_npu.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(nn.Module):
    """NPU-accelerated 2D Convolution layer.

    Applies a 2D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        device (optional): Allocation device. Defaults to None.
        dtype (optional): Desired tensor dtype. Defaults to None.

    Examples:
        >>> from intel_npu_acceleration.nn import Conv2d
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> out = conv(x)
        >>> print(out.shape)
        torch.Size([1, 16, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution kernel parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute 2D convolution on NPU."""
        return F_npu.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm(nn.Module):
    r"""NPU-accelerated Root Mean Square Normalization (RMSNorm) layer.

    Used widely across modern Large Language Model (LLM) architectures such as Llama,
    Mistral, Qwen, and DeepSeek.

    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

    Args:
        hidden_size (int): The dimension of the input feature space.
        eps (float, optional): Small epsilon value added to denominator to avoid division by zero. Defaults to 1e-6.
        device (optional): Target allocation device. Defaults to None.
        dtype (optional): Desired tensor dtype. Defaults to None.

    Examples:
        >>> from intel_npu_acceleration.nn import RMSNorm
        >>> norm = RMSNorm(4096)
        >>> x = torch.randn(2, 512, 4096)
        >>> out = norm(x)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm on NPU."""
        return F_npu.rmsnorm(x, self.weight, self.eps)


class LayerNorm(nn.Module):
    """NPU-accelerated Layer Normalization layer.

    Applies Layer Normalization over a mini-batch of inputs with learnable affine weights.

    Args:
        normalized_shape (Union[int, List[int], torch.Size]): Input shape from an expected input of size.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): A boolean value that when set to True, enables learnable affine parameters. Defaults to True.
        device (optional): Target allocation device. Defaults to None.
        dtype (optional): Desired tensor dtype. Defaults to None.

    Examples:
        >>> from intel_npu_acceleration.nn import LayerNorm
        >>> norm = LayerNorm(768)
        >>> x = torch.randn(4, 128, 768)
        >>> out = norm(x)
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        factory_kwargs = {"device": device, "dtype": dtype}

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm on NPU."""
        return F_npu.layer_norm(
            input, list(self.normalized_shape), self.weight, self.bias, self.eps
        )


class Embedding(nn.Module):
    """NPU-accelerated Embedding lookup table.

    Used to store token or position embeddings and retrieve them using integer indices.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        device (optional): Target allocation device. Defaults to None.
        dtype (optional): Desired tensor dtype. Defaults to None.

    Examples:
        >>> from intel_npu_acceleration.nn import Embedding
        >>> emb = Embedding(32000, 4096)
        >>> input_ids = torch.randint(0, 32000, (2, 64))
        >>> tokens = emb(input_ids)
        >>> print(tokens.shape)
        torch.Size([2, 64, 4096])
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embedding table with normal distribution."""
        nn.init.normal_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform embedding gather on NPU."""
        return F_npu.embedding(input, self.weight)


class MSELoss(nn.Module):
    """NPU-accelerated Mean Squared Error Loss layer.

    Measures the element-wise mean squared error (squared L2 norm) between
    predictions and targets.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.

    Examples:
        >>> from intel_npu_acceleration.nn import MSELoss
        >>> criterion = MSELoss()
        >>> pred = torch.randn(4, 10, requires_grad=True)
        >>> target = torch.randn(4, 10)
        >>> loss = criterion(pred, target)
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss on NPU."""
        return F_npu.mse_loss(input, target, reduction=self.reduction)

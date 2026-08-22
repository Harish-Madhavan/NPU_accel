"""
Intel NPU Custom Neural Network Modules and Quantization.
"""

from .stateful_kv import NPUStatefulKVCache
from .quantized import quantize
from .modules import (
    Linear,
    Conv2d,
    RMSNorm,
    LayerNorm,
    Embedding,
    MSELoss,
)

__all__ = [
    "NPUStatefulKVCache",
    "quantize",
    "Linear",
    "Conv2d",
    "RMSNorm",
    "LayerNorm",
    "Embedding",
    "MSELoss",
]

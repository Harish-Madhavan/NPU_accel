"""
Intel NPU Custom Neural Network Modules and Quantization.
"""

from .modules import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Embedding,
    L1Loss,
    LayerNorm,
    Linear,
    MSELoss,
    RMSNorm,
)
from .quantized import quantize
from .stateful_kv import NPUStatefulKVCache

__all__ = [
    "NPUStatefulKVCache",
    "quantize",
    "Linear",
    "Conv2d",
    "RMSNorm",
    "LayerNorm",
    "Embedding",
    "MSELoss",
    "CrossEntropyLoss",
    "L1Loss",
    "BCEWithLogitsLoss",
]

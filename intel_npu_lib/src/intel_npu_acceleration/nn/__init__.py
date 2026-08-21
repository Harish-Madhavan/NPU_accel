"""
Intel NPU Custom Neural Network Modules and Quantization.
"""

from .stateful_kv import NPUStatefulKVCache
from .quantized import quantize

__all__ = [
    "NPUStatefulKVCache",
    "quantize",
]

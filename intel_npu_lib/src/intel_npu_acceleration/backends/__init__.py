"""Backend registration for Intel NPU as proper PyTorch device."""

from .npu_backend import register_pytorch_backend

__all__ = ["register_pytorch_backend"]

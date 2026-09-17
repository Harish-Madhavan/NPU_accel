import operator
from typing import Any

import torch

import intel_npu_acceleration.functional as npu_func

__all__ = ["OpRegistry"]

# Supported Operation Sets
_SUPPORTED_FUNCTIONS = {
    torch.add, torch.sub, torch.mul, torch.div, torch.neg,
    npu_func.add, npu_func.sub, npu_func.mul, npu_func.div, npu_func.neg,
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.neg, operator.getitem, getattr,
    torch.matmul, torch.nn.functional.linear, torch.nn.functional.scaled_dot_product_attention,
    npu_func.matmul, npu_func.linear, npu_func.quantized_linear,
    torch.sigmoid, torch.tanh, torch.clamp, torch.relu,
    torch.nn.functional.relu, torch.nn.functional.gelu, torch.nn.functional.silu,
    torch.nn.functional.relu6, torch.nn.functional.leaky_relu, torch.nn.functional.hardtanh,
    torch.nn.functional.hardsigmoid, torch.nn.functional.hardswish,
    torch.nn.functional.softmax, torch.softmax,
    npu_func.relu, npu_func.gelu, npu_func.silu, npu_func.hardsigmoid, npu_func.hardswish, npu_func.softmax,
    torch.transpose, torch.reshape, torch.squeeze, torch.unsqueeze, torch.cat, torch.stack, torch.mean,
    torch.sin, torch.cos, torch.rsqrt, torch.pow, torch.where, torch.exp, torch.sqrt, torch.abs,
    torch.flatten, torch.clone, torch.triu, torch.arange,
    npu_func.transpose, npu_func.reshape, npu_func.squeeze, npu_func.unsqueeze, npu_func.cat,
    npu_func.stack, npu_func.mean, npu_func.index_select, npu_func.zeros, npu_func.ones, npu_func.full,
    npu_func.identity, npu_func.dropout,
    torch.nn.functional.layer_norm, torch.nn.functional.conv2d, torch.nn.functional.max_pool2d,
    torch.nn.functional.avg_pool2d, torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.embedding, torch.nn.functional.dropout, torch.nn.functional.batch_norm,
    torch.nn.functional.mse_loss, torch.nn.functional.cross_entropy,
    torch.nn.functional.l1_loss, torch.nn.functional.binary_cross_entropy_with_logits,
    npu_func.conv2d, npu_func.layer_norm, npu_func.rmsnorm, npu_func.max_pool2d,
    npu_func.update_kv_cache, npu_func.embedding, npu_func.scaled_dot_product_attention,
    npu_func.mse_loss, npu_func.cross_entropy_loss, npu_func.l1_loss, npu_func.bce_with_logits_loss,
}

_SUPPORTED_METHODS = {
    "add", "sub", "mul", "matmul", "div", "neg", "transpose", "reshape", "squeeze", "unsqueeze", "view", "to",
    "sigmoid", "tanh", "clamp", "relu", "sin", "cos", "rsqrt", "pow", "flatten", "permute", "chunk", "split",
    "contiguous", "clone", "softmax", "mean", "sum", "abs", "exp", "sqrt", "type_as", "float", "half", "int", "long"
}

_SUPPORTED_MODULES = {
    torch.nn.Linear, torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.LayerNorm, torch.nn.Conv2d, torch.nn.MaxPool2d,
    torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.BatchNorm2d, torch.nn.Dropout, torch.nn.Embedding,
    torch.nn.Identity, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU6, torch.nn.LeakyReLU, torch.nn.Hardtanh,
    torch.nn.Hardsigmoid, torch.nn.Hardswish, torch.nn.MSELoss, torch.nn.CrossEntropyLoss,
    torch.nn.L1Loss, torch.nn.BCEWithLogitsLoss,
    npu_func.NPUStatefulKVCache
}


class OpRegistry:
    """Lightweight set-based registry for checking NPU offload eligibility."""

    _functions = _SUPPORTED_FUNCTIONS
    _methods = _SUPPORTED_METHODS
    _modules = _SUPPORTED_MODULES

    @classmethod
    def is_function_supported(cls, target: Any) -> bool:
        """Return True if the function target can be offloaded to NPU."""
        return target in cls._functions

    @classmethod
    def is_method_supported(cls, name: str) -> bool:
        """Return True if the method name can be offloaded to NPU."""
        return name in cls._methods

    @classmethod
    def is_module_supported(cls, module_type: Any) -> bool:
        """Return True if the module type can be offloaded to NPU."""
        return module_type in cls._modules

    # Backward-compat shims — return True/None as before
    @classmethod
    def get_function(cls, target: Any) -> bool | None:
        return True if cls.is_function_supported(target) else None

    @classmethod
    def get_method(cls, name: str) -> bool | None:
        return True if cls.is_method_supported(name) else None

    @classmethod
    def get_module(cls, module_type: Any) -> bool | None:
        return True if cls.is_module_supported(module_type) else None

    @classmethod
    def is_supported_function(cls, target: Any) -> bool:
        return cls.is_function_supported(target)

    @classmethod
    def is_supported_method(cls, name: str) -> bool:
        return cls.is_method_supported(name)

    @classmethod
    def is_supported_module(cls, module_type: Any) -> bool:
        return cls.is_module_supported(module_type)

"""
Quantized Neural Network Layers and Utilities for Intel NPU.
"""

import torch
import torch.nn as nn


def quantize(model: nn.Module) -> nn.Module:
    """
    Quantize the weights of all `torch.nn.Linear` layers in the model to INT8 precision.
    Compresses model footprint by up to 2x and prepares weights for fast NPU execution.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                scale = w.abs().max() / 127.0
                quantized_w = (w / scale).round().clamp(-128, 127).to(torch.int8)

                module.weight.requires_grad = False
                module.weight.data = quantized_w
                setattr(module, "weight_scale", scale)
    return model

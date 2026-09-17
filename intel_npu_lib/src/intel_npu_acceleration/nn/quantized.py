"""
Quantized Neural Network Layers and Utilities for Intel NPU.
"""

import torch
import torch.nn as nn


def quantize(
    model: nn.Module,
    mode: str = "int8",
    granularity: str = "per-channel",
) -> nn.Module:
    """
    Quantize all `torch.nn.Linear` layers in the model for Intel NPU acceleration.

    Supports:
      - mode="int8": 8-bit symmetric quantization (per-channel or per-tensor).
      - mode="int4": 4-bit asymmetric quantization packed as 2 nibbles per uint8 byte.
      - granularity: "per-channel" (default, row-wise scaling) or "per-tensor" (scalar scaling).

    Args:
        model (nn.Module): Model containing Linear layers to quantize.
        mode (str, optional): Target precision ('int8' or 'int4'). Defaults to 'int8'.
        granularity (str, optional): Scale granularity ('per-channel' or 'per-tensor'). Defaults to 'per-channel'.

    Returns:
        nn.Module: Model with quantized weights updated in-place.
    """
    if mode not in ["int8", "int4"]:
        raise ValueError(f"Unsupported quantization mode: '{mode}'. Expected 'int8' or 'int4'.")
    if granularity not in ["per-channel", "per-tensor"]:
        raise ValueError(f"Unsupported granularity: '{granularity}'. Expected 'per-channel' or 'per-tensor'.")

    with torch.no_grad():
        for _name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                c_out, c_in = w.shape

                if mode == "int8":
                    if granularity == "per-channel":
                        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                    else:
                        scale = w.abs().max().clamp(min=1e-6) / 127.0
                    quantized_w = (w / scale).round().clamp(-128, 127).to(torch.int8)

                    module.weight.requires_grad = False
                    module.weight.data = quantized_w
                    module.weight_scale = scale
                    module.quantization_mode = "int8"
                    module.quantization_granularity = granularity

                elif mode == "int4":
                    if c_in % 2 != 0:
                        raise ValueError(f"INT4 packing requires in_features to be even, got {c_in}")
                    if granularity == "per-channel":
                        w_min = w.amin(dim=-1, keepdim=True)
                        w_max = w.amax(dim=-1, keepdim=True)
                    else:
                        w_min = w.min()
                        w_max = w.max()
                    scale = (w_max - w_min).clamp(min=1e-6) / 15.0
                    zero_point = (-w_min / scale).round().clamp(0, 15).to(torch.int8)

                    q_w = ((w - w_min) / scale).round().clamp(0, 15).to(torch.uint8)
                    q_even = q_w[:, 0::2]
                    q_odd = q_w[:, 1::2]
                    packed = q_even + q_odd * 16

                    module.weight.requires_grad = False
                    module.weight.data = packed
                    module.weight_scale = scale
                    module.weight_zero_point = zero_point
                    module.quantization_mode = "int4"
                    module.quantization_granularity = granularity

    return model


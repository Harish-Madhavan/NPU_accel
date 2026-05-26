import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder


def _to_list(val, n=2):
    if isinstance(val, int):
        return [val] * n
    return list(val)


def _get_partial_shape(node_or_output):
    if isinstance(node_or_output, ov.Output):
        return node_or_output.get_partial_shape()
    return node_or_output.get_output_partial_shape(0)


@OpRegistry.register_function(torch.nn.functional.conv2d)
def convert_conv2d(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    bias = (
        builder.get_input_or_constant(args[2])
        if len(args) > 2
        else kwargs.get("bias", None)
    )

    stride = kwargs.get("stride", args[3] if len(args) > 3 else 1)
    padding = kwargs.get("padding", args[4] if len(args) > 4 else 0)
    dilation = kwargs.get("dilation", args[5] if len(args) > 5 else 1)
    groups = kwargs.get("groups", args[6] if len(args) > 6 else 1)

    ov_strides = np.array(_to_list(stride), dtype=np.int64)
    ov_pads_begin = np.array(_to_list(padding), dtype=np.int64)
    ov_pads_end = np.array(_to_list(padding), dtype=np.int64)
    ov_dilations = np.array(_to_list(dilation), dtype=np.int64)

    if groups == 1:
        conv = ops.convolution(
            inp, weight, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations
        )
    else:
        w_shape = _get_partial_shape(weight)
        c_out = w_shape[0].get_length()
        c_in_g = w_shape[1].get_length()
        kH = w_shape[2].get_length()
        kW = w_shape[3].get_length()
        new_shape = ops.constant(
            np.array([groups, c_out // groups, c_in_g, kH, kW], dtype=np.int64)
        )
        weight_grouped = ops.reshape(weight, new_shape, special_zero=False)
        conv = ops.group_convolution(
            inp, weight_grouped, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations
        )

    res = conv
    if bias is not None:
        axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
        bias_4d = ops.unsqueeze(bias, axes)
        res = ops.add(conv, bias_4d)

    return res


@OpRegistry.register_module(torch.nn.Conv2d)
def convert_conv2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)

    if w_const.get_element_type().get_type_name() in ["i8", "u8"]:
        w_const = ops.convert(w_const, destination_type=np.float16)
        if hasattr(submod, "weight_scale"):
            scale = getattr(submod, "weight_scale")
            if isinstance(scale, torch.Tensor):
                scale_val = scale.detach().cpu().numpy()
            else:
                scale_val = float(scale)
            scale_node = ops.constant(scale_val, dtype=np.float32)
            w_const = ops.multiply(w_const, scale_node)

    ov_strides = np.array(_to_list(submod.stride), dtype=np.int64)
    ov_pads_begin = np.array(_to_list(submod.padding), dtype=np.int64)
    ov_pads_end = np.array(_to_list(submod.padding), dtype=np.int64)
    ov_dilations = np.array(_to_list(submod.dilation), dtype=np.int64)
    groups = submod.groups

    if groups == 1:
        conv = ops.convolution(
            inp, w_const, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations
        )
    else:
        c_out, c_in_g, kH, kW = submod.weight.shape
        new_shape = ops.constant(
            np.array([groups, c_out // groups, c_in_g, kH, kW], dtype=np.int64)
        )
        weight_grouped = ops.reshape(w_const, new_shape, special_zero=False)
        conv = ops.group_convolution(
            inp, weight_grouped, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations
        )

    res = conv
    if submod.bias is not None:
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
        bias_4d = ops.unsqueeze(b_const, axes)
        res = ops.add(conv, bias_4d)

    return res


@OpRegistry.register_function(torch.nn.functional.avg_pool2d)
def convert_avg_pool2d(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    kernel_size = kwargs.get("kernel_size", args[1] if len(args) > 1 else None)
    stride = kwargs.get("stride", args[2] if len(args) > 2 else None)
    padding = kwargs.get("padding", args[3] if len(args) > 3 else 0)
    ceil_mode = kwargs.get("ceil_mode", args[4] if len(args) > 4 else False)
    count_include_pad = kwargs.get(
        "count_include_pad", args[5] if len(args) > 5 else True
    )

    if kernel_size is None:
        raise ValueError("kernel_size must be provided for avg_pool2d")

    if stride is None:
        stride = kernel_size

    k_list = _to_list(kernel_size)
    s_list = _to_list(stride)
    p_list = _to_list(padding)

    ov_strides = np.array(s_list, dtype=np.int64)
    ov_pads_begin = np.array(p_list, dtype=np.int64)
    ov_pads_end = np.array(p_list, dtype=np.int64)
    ov_kernel = np.array(k_list, dtype=np.int64)

    rounding_type = "ceil" if ceil_mode else "floor"
    exclude_pad = not count_include_pad

    return ops.avg_pool(
        inp,
        ov_strides,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        exclude_pad=exclude_pad,
        rounding_type=rounding_type,
    )


@OpRegistry.register_module(torch.nn.AvgPool2d)
def convert_avgpool2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])

    k_list = _to_list(submod.kernel_size)
    s_list = _to_list(submod.stride)
    p_list = _to_list(submod.padding)

    ov_strides = np.array(s_list, dtype=np.int64)
    ov_pads_begin = np.array(p_list, dtype=np.int64)
    ov_pads_end = np.array(p_list, dtype=np.int64)
    ov_kernel = np.array(k_list, dtype=np.int64)

    rounding_type = "ceil" if submod.ceil_mode else "floor"
    exclude_pad = not submod.count_include_pad

    return ops.avg_pool(
        inp,
        ov_strides,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        exclude_pad=exclude_pad,
        rounding_type=rounding_type,
    )


@OpRegistry.register_function(torch.nn.functional.max_pool2d)
def convert_max_pool2d(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    kernel_size = kwargs.get("kernel_size", args[1] if len(args) > 1 else None)
    stride = kwargs.get("stride", args[2] if len(args) > 2 else None)
    padding = kwargs.get("padding", args[3] if len(args) > 3 else 0)
    dilation = kwargs.get("dilation", args[4] if len(args) > 4 else 1)
    ceil_mode = kwargs.get("ceil_mode", args[5] if len(args) > 5 else False)

    if kernel_size is None:
        raise ValueError("kernel_size must be provided for max_pool2d")

    if stride is None:
        stride = kernel_size

    k_list = _to_list(kernel_size)
    s_list = _to_list(stride)
    p_list = _to_list(padding)
    d_list = _to_list(dilation)

    ov_strides = np.array(s_list, dtype=np.int64)
    ov_pads_begin = np.array(p_list, dtype=np.int64)
    ov_pads_end = np.array(p_list, dtype=np.int64)
    ov_kernel = np.array(k_list, dtype=np.int64)
    ov_dilations = np.array(d_list, dtype=np.int64)

    rounding_type = "ceil" if ceil_mode else "floor"

    res = ops.max_pool(
        inp,
        ov_strides,
        ov_dilations,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        rounding_type=rounding_type,
    )
    return res.output(0)


@OpRegistry.register_module(torch.nn.MaxPool2d)
def convert_maxpool2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])

    k_list = _to_list(submod.kernel_size)
    s_list = _to_list(submod.stride)
    p_list = _to_list(submod.padding)
    d_list = _to_list(submod.dilation)

    ov_strides = np.array(s_list, dtype=np.int64)
    ov_pads_begin = np.array(p_list, dtype=np.int64)
    ov_pads_end = np.array(p_list, dtype=np.int64)
    ov_kernel = np.array(k_list, dtype=np.int64)
    ov_dilations = np.array(d_list, dtype=np.int64)

    rounding_type = "ceil" if submod.ceil_mode else "floor"

    res = ops.max_pool(
        inp,
        ov_strides,
        ov_dilations,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        rounding_type=rounding_type,
    )
    return res.output(0)


@OpRegistry.register_module(torch.nn.AdaptiveAvgPool2d)
def convert_adaptive_avg_pool2d(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    output_size = _to_list(submod.output_size)
    if output_size == [1, 1]:
        rank = _get_partial_shape(inp).rank.get_length()
        axes = ops.constant(np.array([rank - 2, rank - 1]), dtype=np.int64)
        return ops.reduce_mean(inp, axes, keep_dims=True)
    else:
        raise NotImplementedError(
            "AdaptiveAvgPool2d only supported for output_size=(1,1) (Global Pool)"
        )


@OpRegistry.register_module(torch.nn.Identity)
@OpRegistry.register_function(torch.nn.functional.dropout)
@OpRegistry.register_module(torch.nn.Dropout, torch.nn.Dropout2d)
def convert_identity(builder: OVGraphBuilder, node, *args, **kwargs):
    if node.op == "call_module":
        fx_args = args[1]
    else:
        fx_args = args[0]

    if len(fx_args) > 0:
        return builder.get_input_or_constant(fx_args[0])
    return builder.get_input_or_constant(node.args[0])

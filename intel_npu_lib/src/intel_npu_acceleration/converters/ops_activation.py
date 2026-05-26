import torch
import openvino.opset13 as ops
import numpy as np
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder
from .. import relu as npu_relu_func
from .. import gelu as npu_gelu_func
from .. import silu as npu_silu_func


@OpRegistry.register_function(torch.relu, torch.nn.functional.relu, npu_relu_func)
def convert_relu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.relu(inp)


@OpRegistry.register_module(torch.nn.ReLU)
def convert_relu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.relu(inp)


@OpRegistry.register_function(torch.nn.functional.gelu, npu_gelu_func)
def convert_gelu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    approx = kwargs.get("approximate", "none")
    mode = "erf" if approx == "none" else "tanh"
    return ops.gelu(inp, approximation_mode=mode)


@OpRegistry.register_module(torch.nn.GELU)
def convert_gelu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    approx = submod.approximate
    mode = "erf" if approx == "none" else "tanh"
    return ops.gelu(inp, approximation_mode=mode)


@OpRegistry.register_function(torch.nn.functional.silu, npu_silu_func)
def convert_silu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.swish(inp)


@OpRegistry.register_module(torch.nn.SiLU)
def convert_silu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.swish(inp)


@OpRegistry.register_function(torch.nn.functional.hardsigmoid)
def convert_hardsigmoid(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    alpha = ops.constant(1.0 / 6.0, dtype=np.float32)
    beta = ops.constant(0.5, dtype=np.float32)
    return ops.hard_sigmoid(inp, alpha, beta)


@OpRegistry.register_module(torch.nn.Hardsigmoid)
def convert_hardsigmoid_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    alpha = ops.constant(1.0 / 6.0, dtype=np.float32)
    beta = ops.constant(0.5, dtype=np.float32)
    return ops.hard_sigmoid(inp, alpha, beta)


@OpRegistry.register_function(torch.nn.functional.hardswish)
def convert_hardswish(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.hswish(inp)


@OpRegistry.register_module(torch.nn.Hardswish)
def convert_hardswish_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.hswish(inp)


@OpRegistry.register_function(torch.sigmoid, torch.nn.functional.sigmoid)
@OpRegistry.register_method("sigmoid")
def convert_sigmoid(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sigmoid(inp)


@OpRegistry.register_module(torch.nn.Sigmoid)
def convert_sigmoid_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sigmoid(inp)


@OpRegistry.register_function(torch.tanh, torch.nn.functional.tanh)
@OpRegistry.register_method("tanh")
def convert_tanh(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.tanh(inp)


@OpRegistry.register_module(torch.nn.Tanh)
def convert_tanh_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.tanh(inp)


@OpRegistry.register_function(torch.nn.functional.relu6)
@OpRegistry.register_method("relu6")
def convert_relu6(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.clamp(inp, 0.0, 6.0)


@OpRegistry.register_module(torch.nn.ReLU6)
def convert_relu6_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.clamp(inp, 0.0, 6.0)


@OpRegistry.register_function(torch.nn.functional.leaky_relu)
@OpRegistry.register_method("leaky_relu")
def convert_leaky_relu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    slope = kwargs.get("negative_slope", args[1] if len(args) > 1 else 0.01)
    slope_node = ops.constant(slope, dtype=np.float32)
    return ops.prelu(inp, slope_node)


@OpRegistry.register_module(torch.nn.LeakyReLU)
def convert_leaky_relu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    slope = submod.negative_slope
    slope_node = ops.constant(slope, dtype=np.float32)
    return ops.prelu(inp, slope_node)


@OpRegistry.register_function(torch.softmax, torch.nn.functional.softmax)
def convert_softmax(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
    return ops.softmax(inp, axis=dim)


@OpRegistry.register_module(torch.nn.Softmax)
def convert_softmax_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    dim = submod.dim if submod.dim is not None else -1
    return ops.softmax(builder.get_input_or_constant(args[0]), axis=dim)


@OpRegistry.register_function(torch.nn.functional.log_softmax)
@OpRegistry.register_module(torch.nn.LogSoftmax)
def convert_log_softmax(builder: OVGraphBuilder, node, *args, **kwargs):
    if isinstance(args[0], torch.nn.Module):
        inp = builder.get_input_or_constant(args[1][0])
        dim = args[0].dim if args[0].dim is not None else -1
    else:
        inp = builder.get_input_or_constant(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)

    softmax = ops.softmax(inp, axis=dim)
    return ops.log(softmax)

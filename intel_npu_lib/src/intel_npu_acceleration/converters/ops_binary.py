import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np
import operator
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder


@OpRegistry.register_function(torch.add, operator.add)
def convert_add(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    return ops.add(inp0, inp1)


@OpRegistry.register_function(torch.sub, operator.sub)
def convert_sub(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    return ops.subtract(inp0, inp1)


@OpRegistry.register_function(torch.mul, operator.mul)
def convert_mul(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    return ops.multiply(inp0, inp1)


@OpRegistry.register_function(torch.div, operator.truediv)
def convert_div(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])

    inp0, inp1 = builder.align_types(inp0, inp1)
    if not str(inp0.get_element_type()).startswith("f"):
        inp0 = ops.convert(inp0, destination_type=np.float32)
        inp1 = ops.convert(inp1, destination_type=np.float32)

    return ops.divide(inp0, inp1)


@OpRegistry.register_function(operator.floordiv)
def convert_floordiv(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    div = ops.divide(inp0, inp1)
    res = ops.floor(div)
    return ops.convert(res, destination_type=np.int64)


@OpRegistry.register_function(torch.pow, operator.pow)
@OpRegistry.register_method("pow")
def convert_pow(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    exponent = builder.get_input_or_constant(args[1])
    inp, exponent = builder.align_types(inp, exponent)
    return ops.power(inp, exponent)


@OpRegistry.register_function(torch.neg, operator.neg)
@OpRegistry.register_method("neg")
def convert_neg(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.negative(inp)


@OpRegistry.register_function(torch.sin)
def convert_sin(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sin(inp)


@OpRegistry.register_function(torch.cos)
def convert_cos(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.cos(inp)


@OpRegistry.register_function(torch.sqrt)
def convert_sqrt(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sqrt(inp)


@OpRegistry.register_function(torch.rsqrt)
def convert_rsqrt(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.power(inp, ops.constant([-0.5], dtype=np.float32))


@OpRegistry.register_function(torch.log)
def convert_log(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.log(inp)


@OpRegistry.register_function(torch.exp)
def convert_exp(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.exp(inp)


@OpRegistry.register_function(torch.abs)
def convert_abs(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.abs(inp)


@OpRegistry.register_function(torch.where)
def convert_where(builder: OVGraphBuilder, node, args, kwargs):
    cond = builder.get_input_or_constant(args[0])
    x = builder.get_input_or_constant(args[1])
    y = builder.get_input_or_constant(args[2])
    x, y = builder.align_types(x, y)
    return ops.select(cond, x, y)


@OpRegistry.register_function(torch.clamp, torch.nn.functional.hardtanh)
@OpRegistry.register_method("clamp")
def convert_clamp(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    min_val = kwargs.get("min", args[1] if len(args) > 1 else None)
    max_val = kwargs.get("max", args[2] if len(args) > 2 else None)

    if min_val is not None:
        min_node = builder.get_input_or_constant(min_val)
        inp, min_node = builder.align_types(inp, min_node)
        inp = ops.maximum(inp, min_node)

    if max_val is not None:
        max_node = builder.get_input_or_constant(max_val)
        inp, max_node = builder.align_types(inp, max_node)
        inp = ops.minimum(inp, max_node)

    return inp


@OpRegistry.register_module(torch.nn.Hardtanh)
def convert_hardtanh_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    min_val = submod.min_val
    max_val = submod.max_val

    if min_val is not None:
        min_node = builder.get_input_or_constant(min_val)
        inp, min_node = builder.align_types(inp, min_node)
        inp = ops.maximum(inp, min_node)

    if max_val is not None:
        max_node = builder.get_input_or_constant(max_val)
        inp, max_node = builder.align_types(inp, max_node)
        inp = ops.minimum(inp, max_node)

    return inp


@OpRegistry.register_function(torch.full)
def convert_full(builder: OVGraphBuilder, node, args, kwargs):
    size = args[0]
    fill_value = args[1]
    dtype = kwargs.get("dtype", torch.float32)

    np_dtype = np.float32
    if dtype == torch.int64:
        np_dtype = np.int64

    if isinstance(size, (tuple, list)):
        shape_nodes = []
        for s in size:
            if isinstance(s, int):
                shape_nodes.append(ops.constant([s], dtype=np.int64))
            else:
                v = builder.get_input_or_constant(s)
                shape_nodes.append(
                    ops.reshape(
                        v, ops.constant([1], dtype=np.int64), special_zero=False
                    )
                )
        shape_node = ops.concat(shape_nodes, axis=0)
    else:
        shape_node = builder.get_input_or_constant(size)

    val_node = ops.constant(np.array([fill_value], dtype=np_dtype))
    return ops.broadcast(val_node, shape_node)


@OpRegistry.register_function(torch.zeros, torch.zeros_like)
def convert_zeros(builder: OVGraphBuilder, node, args, kwargs):
    if node.target == torch.zeros_like:
        ref = builder.get_input_or_constant(args[0])
        shape_node = ops.shape_of(ref)
        dtype = ref.get_element_type()
    else:
        size = args[0]
        shape_node = builder.get_input_or_constant(size)
        dtype = np.float32  # Default

    val_node = ops.constant(0.0, dtype=dtype)
    return ops.broadcast(val_node, shape_node)


@OpRegistry.register_function(torch.ones, torch.ones_like)
def convert_ones(builder: OVGraphBuilder, node, args, kwargs):
    if node.target == torch.ones_like:
        ref = builder.get_input_or_constant(args[0])
        shape_node = ops.shape_of(ref)
        dtype = ref.get_element_type()
    else:
        size = args[0]
        shape_node = builder.get_input_or_constant(size)
        dtype = np.float32

    val_node = ops.constant(1.0, dtype=dtype)
    return ops.broadcast(val_node, shape_node)


@OpRegistry.register_function(torch.arange)
def convert_arange(builder: OVGraphBuilder, node, args, kwargs):
    if len(args) == 1:
        start = ops.constant(0, dtype=np.float32)
        stop = builder.get_input_or_constant(args[0])
        step = ops.constant(1, dtype=np.float32)
    elif len(args) == 2:
        start = builder.get_input_or_constant(args[0])
        stop = builder.get_input_or_constant(args[1])
        step = ops.constant(1, dtype=start.get_element_type())
    else:
        start = builder.get_input_or_constant(args[0])
        stop = builder.get_input_or_constant(args[1])
        step = builder.get_input_or_constant(args[2])

    start, stop = builder.align_types(start, stop)
    start, step = builder.align_types(start, step)

    return ops.range(start, stop, step, output_type=start.get_element_type())


@OpRegistry.register_function(torch.clone)
@OpRegistry.register_method("clone")
def convert_clone(builder: OVGraphBuilder, node, args, kwargs):
    return builder.get_input_or_constant(args[0])


@OpRegistry.register_function(torch.matmul, torch.mm, operator.matmul)
@OpRegistry.register_method("matmul")
def convert_matmul(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])

    # Check for INT8/UINT8 operands and promote to float16 to trigger NPU INT8 acceleration
    t0 = inp0.get_element_type().get_type_name()
    t1 = inp1.get_element_type().get_type_name()

    if t0 in ["i8", "u8"]:
        inp0 = ops.convert(inp0, destination_type=np.float16)
    if t1 in ["i8", "u8"]:
        inp1 = ops.convert(inp1, destination_type=np.float16)

    inp0, inp1 = builder.align_types(inp0, inp1)

    return ops.matmul(inp0, inp1, transpose_a=False, transpose_b=False)

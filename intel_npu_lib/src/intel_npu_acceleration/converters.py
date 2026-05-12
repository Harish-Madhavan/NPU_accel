import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np
import operator
import builtins
from .registry import OpRegistry
from .graph_builder import OVGraphBuilder
from . import transpose as npu_transpose_func
from . import reshape as npu_reshape_func
from . import rmsnorm as npu_rmsnorm_func
from . import linear as npu_linear_func
from . import relu as npu_relu_func
from . import gelu as npu_gelu_func
from . import silu as npu_silu_func


def _to_list(val, n=2):
    if isinstance(val, int):
        return [val] * n
    return list(val)


# --- Converters ---


@OpRegistry.register_function(torch.nn.functional.scaled_dot_product_attention)
def convert_sdpa(builder: OVGraphBuilder, node, args, kwargs):
    query = builder.get_input_or_constant(args[0])
    key = builder.get_input_or_constant(args[1])
    value = builder.get_input_or_constant(args[2])

    attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
    is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
    scale = kwargs.get("scale", args[6] if len(args) > 6 else None)

    ov_mask = None
    if attn_mask is not None:
        ov_mask = builder.get_input_or_constant(attn_mask)

    if scale is None:
        # Default scale: 1 / sqrt(query.size(-1))
        q_shape = query.get_output_partial_shape(0)
        head_dim = q_shape[-1].get_length()
        scale_val = 1.0 / np.sqrt(head_dim)
        scale_node = ops.constant(scale_val, dtype=np.float32)
    else:
        scale_node = builder.get_input_or_constant(scale)
        if not isinstance(scale_node, ov.Node):
            scale_node = ops.constant(scale_node, dtype=np.float32)

    return ops.scaled_dot_product_attention(
        query, key, value, attention_mask=ov_mask, scale=scale_node, causal=is_causal
    )


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


@OpRegistry.register_function(torch.matmul, torch.mm)
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


@OpRegistry.register_function(torch.relu, torch.nn.functional.relu, npu_relu_func)
def convert_relu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.relu(inp)


@OpRegistry.register_function(torch.nn.functional.gelu, npu_gelu_func)
def convert_gelu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    approx = kwargs.get("approximate", "none")
    mode = "erf" if approx == "none" else "tanh"
    return ops.gelu(inp, approximation_mode=mode)


@OpRegistry.register_function(torch.nn.functional.silu, npu_silu_func)
def convert_silu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.swish(inp)


@OpRegistry.register_function(torch.rsqrt)
def convert_rsqrt(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    exp_node = ops.constant([-0.5], dtype=np.float32)
    inp, exp_node = builder.align_types(inp, exp_node)
    return ops.power(inp, exp_node)


@OpRegistry.register_function(torch.where)
def convert_where(builder: OVGraphBuilder, node, args, kwargs):
    cond = builder.get_input_or_constant(args[0])
    x = builder.get_input_or_constant(args[1])
    y = builder.get_input_or_constant(args[2])
    x, y = builder.align_types(x, y)
    return ops.select(cond, x, y)


@OpRegistry.register_function(torch.clamp, torch.nn.functional.hardtanh)
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


@OpRegistry.register_function(torch.mean)
@OpRegistry.register_method("mean")
def convert_mean(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)

    if dim is None:
        # Global mean: reduce over all axes.
        rank = inp.get_output_partial_shape(0).rank.get_length()
        axes = ops.constant(list(range(rank)), dtype=np.int64)
    elif isinstance(dim, int):
        axes = ops.constant([dim], dtype=np.int64)
    else:
        axes = ops.constant(list(dim), dtype=np.int64)

    return ops.reduce_mean(inp, axes, keep_dims=keepdim)


@OpRegistry.register_function(torch.softmax, torch.nn.functional.softmax)
def convert_softmax(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
    return ops.softmax(inp, axis=dim)


@OpRegistry.register_function(torch.cat)
def convert_cat(builder: OVGraphBuilder, node, args, kwargs):
    tensors_list = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
    ov_inputs = [builder.get_input_or_constant(t) for t in tensors_list]
    return ops.concat(ov_inputs, axis=dim)


@OpRegistry.register_function(torch.stack)
def convert_stack(builder: OVGraphBuilder, node, args, kwargs):
    tensors_list = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
    ov_inputs = [builder.get_input_or_constant(t) for t in tensors_list]

    unsqueezed_inputs = []
    for ov_input in ov_inputs:
        unsqueezed_inputs.append(
            ops.unsqueeze(ov_input, ops.constant(np.array([dim]), dtype=np.int64))
        )

    return ops.concat(unsqueezed_inputs, axis=dim)


@OpRegistry.register_function(torch.transpose, npu_transpose_func)
@OpRegistry.register_method("transpose")
def convert_transpose(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim0 = args[1]
    dim1 = args[2]
    rank = inp.get_output_partial_shape(0).rank.get_length()
    perm = list(range(rank))
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return ops.transpose(inp, perm)


@OpRegistry.register_function(torch.reshape, npu_reshape_func)
@OpRegistry.register_method("reshape", "view")
def convert_reshape(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    # Shape might be tuple in args[1] or varargs in args[1:]
    req_shape = args[1]
    if isinstance(req_shape, (tuple, list)):
        shape_items = req_shape
    else:
        shape_items = args[1:]

    shape_nodes = []
    for item in shape_items:
        if isinstance(item, int):
            shape_nodes.append(ops.constant([item], dtype=np.int64))
        elif isinstance(item, torch.fx.Node):
            val_node = builder.get_input(item.name)
            shape_nodes.append(
                ops.reshape(
                    val_node, ops.constant([1], dtype=np.int64), special_zero=False
                )
            )
        else:
            raise RuntimeError(f"Unknown shape item type: {type(item)}")

    shape_tensor = ops.concat(shape_nodes, axis=0)
    return ops.reshape(inp, shape_tensor, special_zero=False)


@OpRegistry.register_function(npu_rmsnorm_func)
def convert_rmsnorm(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    eps = args[2]

    x_sq = ops.multiply(inp, inp)
    rank = inp.get_output_partial_shape(0).rank.get_length()
    axes = ops.constant([rank - 1], dtype=np.int64)
    mean_sq = ops.reduce_mean(x_sq, axes, keep_dims=True)

    eps_const = ops.constant(eps, dtype=np.float32)
    variance = ops.add(mean_sq, eps_const)
    std_dev = ops.sqrt(variance)
    x_norm = ops.divide(inp, std_dev)
    return ops.multiply(x_norm, weight)


@OpRegistry.register_function(torch.nn.functional.layer_norm)
def convert_layer_norm(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    normalized_shape = args[1]
    weight = builder.get_input_or_constant(kwargs.get("weight", args[2] if len(args) > 2 else None))
    bias = builder.get_input_or_constant(kwargs.get("bias", args[3] if len(args) > 3 else None))
    eps = kwargs.get("eps", args[4] if len(args) > 4 else 1e-5)

    rank = inp.get_output_partial_shape(0).rank.get_length()
    norm_rank = len(normalized_shape)
    axes_list = list(range(rank - norm_rank, rank))
    axes = ops.constant(axes_list, dtype=np.int64)

    mean = ops.reduce_mean(inp, axes, keep_dims=True)
    sub = ops.subtract(inp, mean)
    sq = ops.multiply(sub, sub)
    variance = ops.reduce_mean(sq, axes, keep_dims=True)

    eps_const = ops.constant(eps, dtype=np.float32)
    var_eps = ops.add(variance, eps_const)
    std_dev = ops.sqrt(var_eps)

    x_norm = ops.divide(sub, std_dev)

    if weight is not None:
        x_norm = ops.multiply(x_norm, weight)
    if bias is not None:
        x_norm = ops.add(x_norm, bias)

    return x_norm


@OpRegistry.register_function(builtins.getattr)
def convert_getattr(builder: OVGraphBuilder, node, args, kwargs):
    obj_node = args[0]
    attr_name = args[1]
    if attr_name == "shape":
        if "tensor_meta" in obj_node.meta:
            shape = list(obj_node.meta["tensor_meta"].shape)
            return ops.constant(np.array(shape, dtype=np.int64))
        else:
            raise RuntimeError(f"Cannot get shape for {obj_node.name}: no meta info")
    elif attr_name == "device":
        return ops.constant(np.array([0], dtype=np.int32))
    raise NotImplementedError(f"getattr({attr_name}) not implemented")


@OpRegistry.register_function(torch.full)
def convert_full(builder: OVGraphBuilder, node, args, kwargs):
    size = args[0]
    fill_value = args[1]

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

    val_node = builder.get_input_or_constant(fill_value)
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


@OpRegistry.register_function(operator.setitem)
@OpRegistry.register_method("__setitem__")
def convert_setitem(builder: OVGraphBuilder, node, args, kwargs):
    """
    Handle tensor[indices] = value by decomposing into concat operations.

    Supported index patterns
    ------------------------
    * tensor[i]                  — integer index on dim 0
    * tensor[i, j, ...]          — tuple of integers (selects a single element location)
    * tensor[start:stop]         — single slice on dim 0
    * tensor[i, start:stop, ...] — tuple with one or more slices and/or int indices
    * tensor[:, i, :]            — any combination of the above across multiple dims

    The implementation builds a ScatterNDUpdate graph which correctly handles
    all of the above patterns without positional slice arithmetic.
    """
    target_node_name = args[0].name
    target_ov = builder.get_input(target_node_name)
    indices_raw = args[1]
    value_ov = builder.get_input_or_constant(args[2])

    # Normalise to a tuple
    if not isinstance(indices_raw, tuple):
        indices_raw = (indices_raw,)

    target_ov.get_output_partial_shape(0).rank.get_length()

    # Determine which dims are integer-indexed (scalar select) and which are sliced.
    # We build a begin/end/step for StridedSlice to identify the *destination* region,
    # then use concat (split-insert-split) on each sliced dimension.

    # Fast path: all indices are plain scalars → ScatterNDUpdate is cleanest
    all_int = all(isinstance(i, (int, torch.fx.Node)) for i in indices_raw)
    if all_int:
        # Build an index vector: [i0, i1, ..., iN]
        coord_nodes = []
        for idx in indices_raw:
            if isinstance(idx, int):
                coord_nodes.append(ops.constant([idx], dtype=np.int64))
            else:  # fx.Node
                v = builder.get_input(idx.name)
                v = ops.reshape(v, ops.constant([1], dtype=np.int64), False)
                coord_nodes.append(v)
        # ScatterNDUpdate expects indices shape (1, ndim) for a single element
        coord = ops.concat(coord_nodes, axis=0)  # (ndim,)
        coord = ops.unsqueeze(coord, ops.constant([0], dtype=np.int64))  # (1, ndim)
        # value_ov needs to be shape (1,) for ScatterNDUpdate
        val_flat = ops.reshape(value_ov, ops.constant([1], dtype=np.int64), False)
        res = ops.scatter_nd_update(target_ov, coord, val_flat)
        builder.node_map[target_node_name] = res
        return res

    # General path: at least one dimension is a slice.
    # Strategy: iterate over sliced dimensions and apply split-insert-concat.
    # For integer-indexed dims we delegate the value broadcasting to OV.
    #
    # Collect (dim, slice_obj) pairs in order.
    res = target_ov

    def _make_scalar(v):
        """Return a rank-1 shape node for a scalar-or-node index."""
        if isinstance(v, int):
            return ops.constant([v], dtype=np.int64)
        node_v = builder.get_input(v.name)
        return ops.reshape(node_v, ops.constant([1], dtype=np.int64), False)

    for raw_dim, idx in enumerate(indices_raw):
        actual_dim = (
            raw_dim  # after collapsing any earlier integer dims we stay in rank space
        )

        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop  # None means "to end"

            _make_scalar(start) if not isinstance(start, int) else ops.constant(
                [start], dtype=np.int64
            )
            ax = ops.constant([actual_dim], dtype=np.int64)
            st = ops.constant([1], dtype=np.int64)

            # p1: everything before the slice start
            e1 = (
                ops.constant([start], dtype=np.int64)
                if isinstance(start, int)
                else _make_scalar(start)
            )
            p1 = (
                ops.slice(res, ops.constant([0], dtype=np.int64), e1, st, ax)
                if start != 0
                else None
            )

            # p2: everything after the slice stop
            if stop is None:
                p2 = None
            else:
                dim_size_node = ops.gather(
                    ops.shape_of(res),
                    ops.constant([actual_dim], dtype=np.int64),
                    ops.constant([0], dtype=np.int64),
                )
                dim_size_node = ops.reshape(
                    dim_size_node, ops.constant([1], dtype=np.int64), False
                )
                stop_node = (
                    ops.constant([stop], dtype=np.int64)
                    if isinstance(stop, int)
                    else _make_scalar(stop)
                )
                p2 = ops.slice(res, stop_node, dim_size_node, st, ax)

            parts = [p for p in [p1, value_ov, p2] if p is not None]
            res = ops.concat(parts, axis=actual_dim)

        elif isinstance(idx, int):
            # Integer index: the value tensor doesn't have this dimension;
            # unsqueeze it, replace at position idx, then squeeze it back.
            val_expanded = ops.unsqueeze(
                value_ov, ops.constant([actual_dim], dtype=np.int64)
            )
            ax = ops.constant([actual_dim], dtype=np.int64)
            st = ops.constant([1], dtype=np.int64)
            dim_sz = ops.reshape(
                ops.gather(
                    ops.shape_of(res),
                    ops.constant([actual_dim], dtype=np.int64),
                    ops.constant([0], dtype=np.int64),
                ),
                ops.constant([1], dtype=np.int64),
                False,
            )
            p1 = (
                ops.slice(
                    res,
                    ops.constant([0], dtype=np.int64),
                    ops.constant([idx], dtype=np.int64),
                    st,
                    ax,
                )
                if idx != 0
                else None
            )
            p2 = (
                ops.slice(res, ops.constant([idx + 1], dtype=np.int64), dim_sz, st, ax)
                if idx != -1
                else None
            )
            parts = [p for p in [p1, val_expanded, p2] if p is not None]
            res = ops.concat(parts, axis=actual_dim)

        elif isinstance(idx, torch.fx.Node):
            # Dynamic integer index via FX node
            idx_node = builder.get_input(idx.name)
            idx_1d = ops.reshape(idx_node, ops.constant([1], dtype=np.int64), False)
            val_expanded = ops.unsqueeze(
                value_ov, ops.constant([actual_dim], dtype=np.int64)
            )
            ax = ops.constant([actual_dim], dtype=np.int64)
            st = ops.constant([1], dtype=np.int64)
            dim_sz = ops.reshape(
                ops.gather(
                    ops.shape_of(res),
                    ops.constant([actual_dim], dtype=np.int64),
                    ops.constant([0], dtype=np.int64),
                ),
                ops.constant([1], dtype=np.int64),
                False,
            )
            idx_plus_1 = ops.add(idx_1d, ops.constant([1], dtype=np.int64))
            p1 = ops.slice(res, ops.constant([0], dtype=np.int64), idx_1d, st, ax)
            p2 = ops.slice(res, idx_plus_1, dim_sz, st, ax)
            parts = [p for p in [p1, val_expanded, p2] if p is not None]
            res = ops.concat(parts, axis=actual_dim)

    builder.node_map[target_node_name] = res
    return res






@OpRegistry.register_function(torch.index_select)
def convert_index_select(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = args[1]
    index = builder.get_input_or_constant(args[2])

    axis = ops.constant(np.array([dim]), dtype=np.int64)
    return ops.gather(inp, index, axis)


@OpRegistry.register_method("index_copy")
def convert_index_copy(builder: OVGraphBuilder, node, args, kwargs):
    data = builder.get_input_or_constant(args[0])
    dim = builder.get_input_or_constant(args[1])
    indices = builder.get_input_or_constant(args[2])
    updates = builder.get_input_or_constant(args[3])
    
    data, updates = builder.align_types(data, updates)
    
    if not isinstance(dim, ov.Node):
        dim = ops.constant([dim], dtype=np.int64)
    else:
        dim = ops.reshape(dim, ops.constant([1], dtype=np.int64), special_zero=False)
        
    return ops.scatter_update(data, indices, updates, dim)


@OpRegistry.register_function(torch.triu)
def convert_triu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    diagonal = kwargs.get("diagonal", args[1] if len(args) > 1 else 0)

    shape = ops.shape_of(inp)
    rank = inp.get_output_partial_shape(0).rank.get_length()
    if rank < 2:
        return inp

    shape_i32 = ops.convert(shape, destination_type=np.int32)
    h_idx = ops.gather(
        shape_i32,
        ops.constant([rank - 2], dtype=np.int32),
        ops.constant([0], dtype=np.int32),
    )
    w_idx = ops.gather(
        shape_i32,
        ops.constant([rank - 1], dtype=np.int32),
        ops.constant([0], dtype=np.int32),
    )

    h_idx = ops.squeeze(h_idx, ops.constant([0], dtype=np.int32))
    w_idx = ops.squeeze(w_idx, ops.constant([0], dtype=np.int32))

    row_range = ops.range(
        ops.constant(0, dtype=np.int32),
        h_idx,
        ops.constant(1, dtype=np.int32),
        output_type="i32",
    )
    row_range = ops.unsqueeze(row_range, ops.constant([1], dtype=np.int32))

    col_range = ops.range(
        ops.constant(0, dtype=np.int32),
        w_idx,
        ops.constant(1, dtype=np.int32),
        output_type="i32",
    )
    col_range = ops.unsqueeze(col_range, ops.constant([0], dtype=np.int32))

    if isinstance(diagonal, (int, float)):
        diag_const = ops.constant(diagonal, dtype=np.int32)
    else:
        diag_const = builder.get_input_or_constant(diagonal)
        if "32" not in str(diag_const.get_element_type()):
            diag_const = ops.convert(diag_const, destination_type=np.int32)

    threshold = ops.add(row_range, diag_const)
    mask = ops.greater_equal(col_range, threshold)

    zero = ops.constant(0, dtype=inp.get_element_type())
    return ops.select(mask, inp, zero)


@OpRegistry.register_function(operator.getitem)
def convert_getitem(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    idx = args[1]

    if isinstance(idx, int):
        indices = ops.constant([idx], dtype=np.int64)
        axis = ops.constant([0], dtype=np.int64)
        gather = ops.gather(inp, indices, axis)
        return ops.squeeze(gather, axis)

    elif isinstance(idx, (slice, tuple)):
        if isinstance(idx, slice):
            idx = (idx,)

        begin_nodes, end_nodes, strides_nodes = [], [], []
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask = (
            [],
            [],
            [],
            [],
            [],
        )

        for item in idx:
            if item is Ellipsis:
                begin_nodes.append(ops.constant([0], dtype=np.int64))
                end_nodes.append(ops.constant([0], dtype=np.int64))
                strides_nodes.append(ops.constant([1], dtype=np.int64))
                begin_mask.append(0)
                end_mask.append(0)
                ellipsis_mask.append(1)
                new_axis_mask.append(0)
                shrink_axis_mask.append(0)
                continue

            if isinstance(item, slice):
                if item.start is None:
                    start_val = ops.constant([0], dtype=np.int64)
                    begin_mask.append(1)
                else:
                    start_val = ops.reshape(
                        builder.get_input_or_constant(item.start),
                        ops.constant([1], dtype=np.int64),
                        special_zero=False,
                    )
                    begin_mask.append(0)

                if item.stop is None:
                    stop_val = ops.constant([0], dtype=np.int64)
                    end_mask.append(1)
                else:
                    stop_val = ops.reshape(
                        builder.get_input_or_constant(item.stop),
                        ops.constant([1], dtype=np.int64),
                        special_zero=False,
                    )
                    end_mask.append(0)

                if item.step is None:
                    step_val = ops.constant([1], dtype=np.int64)
                else:
                    step_val = ops.reshape(
                        builder.get_input_or_constant(item.step),
                        ops.constant([1], dtype=np.int64),
                        special_zero=False,
                    )

                begin_nodes.append(start_val)
                end_nodes.append(stop_val)
                strides_nodes.append(step_val)
                ellipsis_mask.append(0)
                new_axis_mask.append(0)
                shrink_axis_mask.append(0)

            elif isinstance(item, int):
                start_val = ops.constant([item], dtype=np.int64)
                stop_val = ops.constant([item + 1], dtype=np.int64)
                step_val = ops.constant([1], dtype=np.int64)
                begin_nodes.append(start_val)
                end_nodes.append(stop_val)
                strides_nodes.append(step_val)
                begin_mask.append(0)
                end_mask.append(0)
                ellipsis_mask.append(0)
                new_axis_mask.append(0)
                shrink_axis_mask.append(1)

            elif item is None:
                begin_nodes.append(ops.constant([0], dtype=np.int64))
                end_nodes.append(ops.constant([0], dtype=np.int64))
                strides_nodes.append(ops.constant([1], dtype=np.int64))
                begin_mask.append(0)
                end_mask.append(0)
                ellipsis_mask.append(0)
                new_axis_mask.append(1)
                shrink_axis_mask.append(0)

            else:
                v = builder.get_input(item.name)
                start_val = ops.reshape(
                    v, ops.constant([1], dtype=np.int64), special_zero=False
                )
                one = ops.constant([1], dtype=np.int64)
                stop_val = ops.add(start_val, one)
                step_val = ops.constant([1], dtype=np.int64)
                begin_nodes.append(start_val)
                end_nodes.append(stop_val)
                strides_nodes.append(step_val)
                begin_mask.append(0)
                end_mask.append(0)
                ellipsis_mask.append(0)
                new_axis_mask.append(0)
                shrink_axis_mask.append(1)

        begin_t = ops.concat(begin_nodes, axis=0)
        end_t = ops.concat(end_nodes, axis=0)
        strides_t = ops.concat(strides_nodes, axis=0)

        return ops.strided_slice(
            inp,
            begin_t,
            end_t,
            strides_t,
            begin_mask,
            end_mask,
            new_axis_mask,
            shrink_axis_mask,
            ellipsis_mask,
        )

    elif isinstance(idx, torch.fx.Node):
        idx_node = builder.get_input(idx.name)
        axis = ops.constant([0], dtype=np.int64)
        return ops.gather(inp, idx_node, axis)

    raise NotImplementedError(f"getitem index {type(idx)} not implemented")


# --- Module Converters ---



@OpRegistry.register_module(torch.nn.Embedding)
def convert_embedding_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    indices = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    axis = ops.constant([0], dtype=np.int64)
    return ops.gather(w_const, indices, axis)


@OpRegistry.register_function(torch.nn.functional.embedding)
def convert_embedding_functional(builder: OVGraphBuilder, node, args, kwargs):
    indices = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    axis = ops.constant([0], dtype=np.int64)
    return ops.gather(weight, indices, axis)


@OpRegistry.register_module(torch.nn.ReLU)
def convert_relu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    return ops.relu(builder.get_input_or_constant(args[0]))


@OpRegistry.register_module(torch.nn.GELU)
def convert_gelu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    return ops.gelu(builder.get_input_or_constant(args[0]), approximation_mode="erf")


@OpRegistry.register_module(torch.nn.SiLU)
def convert_silu_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    return ops.swish(builder.get_input_or_constant(args[0]))


@OpRegistry.register_module(torch.nn.Softmax)
def convert_softmax_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    dim = submod.dim if submod.dim is not None else -1
    return ops.softmax(builder.get_input_or_constant(args[0]), axis=dim)


# --- Method Converters ---


@OpRegistry.register_method("contiguous", "type_as", "to")
def convert_noop(builder: OVGraphBuilder, node, args, kwargs):
    return builder.get_input_or_constant(args[0])


@OpRegistry.register_method("float")
def convert_to_float(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.convert(inp, destination_type="f32")


@OpRegistry.register_method("flatten")
def convert_flatten(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    start_dim = args[1] if len(args) > 1 else 0
    end_dim = args[2] if len(args) > 2 else -1

    rank = inp.get_output_partial_shape(0).rank.get_length()
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += rank

    shape_pattern = []
    for i in range(start_dim):
        shape_pattern.append(0)  # Copy
    shape_pattern.append(-1)  # Merge
    for i in range(end_dim + 1, rank):
        shape_pattern.append(0)  # Copy

    return ops.reshape(inp, shape_pattern, special_zero=True)


@OpRegistry.register_method("size")
def convert_size(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    if len(args) > 1:
        dim = args[1]
        shape = ops.shape_of(inp)
        dim_val = ops.gather(
            shape,
            ops.constant([dim], dtype=np.int64),
            ops.constant([0], dtype=np.int64),
        )
        return ops.squeeze(dim_val, ops.constant([0], dtype=np.int64))
    else:
        return ops.shape_of(inp)


@OpRegistry.register_function(torch.nn.functional.conv2d)
def convert_conv2d(builder: OVGraphBuilder, node, args, kwargs):
    # args: input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
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
        # OV group_convolution expects weight shape: (groups, C_out/groups, C_in/groups, kH, kW)
        # The incoming weight is (C_out, C_in/groups, kH, kW) — reshape it.
        w_shape = weight.get_output_partial_shape(0)
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
        bias_node = builder.get_input_or_constant(bias)
        axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
        bias_4d = ops.unsqueeze(bias_node, axes)
        res = ops.add(conv, bias_4d)

    return res


@OpRegistry.register_function(torch.nn.functional.max_pool2d)
def convert_max_pool2d(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    kernel_size = kwargs.get("kernel_size", args[1])
    stride = kwargs.get("stride", args[2] if len(args) > 2 else None)
    padding = kwargs.get("padding", args[3] if len(args) > 3 else 0)
    dilation = kwargs.get("dilation", args[4] if len(args) > 4 else 1)
    ceil_mode = kwargs.get("ceil_mode", args[5] if len(args) > 5 else False)

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

    return ops.max_pool(
        inp,
        ov_strides,
        ov_dilations,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        rounding_type=rounding_type,
    )


@OpRegistry.register_module(torch.nn.Linear)
def convert_linear_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    inp, w_const = builder.align_types(inp, w_const)
    mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)

    if submod.bias is not None:
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        mm, b_const = builder.align_types(mm, b_const)
        return ops.add(mm, b_const)
    return mm


@OpRegistry.register_module(torch.nn.Conv2d)
def convert_conv2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)

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
        # Reshape weight from (C_out, C_in/groups, kH, kW) → (groups, C_out/groups, C_in/groups, kH, kW)
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

    return ops.max_pool(
        inp,
        ov_strides,
        ov_dilations,
        ov_pads_begin,
        ov_pads_end,
        ov_kernel,
        rounding_type=rounding_type,
    )


@OpRegistry.register_module(torch.nn.BatchNorm2d)
def convert_batchnorm2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    eps = submod.eps
    # Axes for broadcasting: unsqueeze over (batch, H, W) → shape (1, C, 1, 1)
    axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)

    if (
        submod.track_running_stats
        and submod.running_mean is not None
        and submod.running_var is not None
    ):
        # --- Inference path: use frozen statistics ---
        mean_const = builder.add_constant(
            f"{node.target}.running_mean", submod.running_mean
        )
        var_const = builder.add_constant(
            f"{node.target}.running_var", submod.running_var
        )
        mean_4d = ops.unsqueeze(mean_const, axes)
        var_4d = ops.unsqueeze(var_const, axes)
    else:
        # --- Training / no-stats path: compute batch mean and variance on-the-fly ---
        # Reduce over (N, H, W), keep C
        reduce_axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
        mean_4d = ops.reduce_mean(inp, reduce_axes, keep_dims=True)  # (1, C, 1, 1)
        diff = ops.subtract(inp, mean_4d)
        var_4d = ops.reduce_mean(
            ops.multiply(diff, diff),
            reduce_axes,
            keep_dims=True,  # (1, C, 1, 1)
        )

    eps_const = ops.constant(eps, dtype=np.float32)

    # Normalize:  (x - mean) / sqrt(var + eps)
    diff = ops.subtract(inp, mean_4d)
    std = ops.sqrt(ops.add(var_4d, eps_const))
    norm = ops.divide(diff, std)

    res = norm
    if submod.affine:
        w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        w_4d = ops.unsqueeze(w_const, axes)
        b_4d = ops.unsqueeze(b_const, axes)
        res = ops.add(ops.multiply(norm, w_4d), b_4d)

    return res


@OpRegistry.register_module(torch.nn.Identity)
@OpRegistry.register_function(torch.nn.functional.dropout)
@OpRegistry.register_module(torch.nn.Dropout, torch.nn.Dropout2d)
def convert_identity(builder: OVGraphBuilder, node, *args, **kwargs):
    # For call_function, args[0] is node.args
    # For call_module, args[1] is node.args
    # We just want the first input to the operation.
    if node.op == "call_module":
        fx_args = args[1]
    else:
        fx_args = args[0]

    if len(fx_args) > 0:
        return builder.get_input_or_constant(fx_args[0])
    return builder.get_input_or_constant(node.args[0])  # fallback


@OpRegistry.register_module(torch.nn.AdaptiveAvgPool2d)
def convert_adaptive_avg_pool2d(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    output_size = _to_list(submod.output_size)
    if output_size == [1, 1]:
        # Global Average Pooling
        rank = inp.get_output_partial_shape(0).rank.get_length()
        axes = ops.constant(np.array([rank - 2, rank - 1]), dtype=np.int64)
        return ops.reduce_mean(inp, axes, keep_dims=True)
    else:
        # Full adaptive pool requires more complex logic or OpenVINO helper
        # For now, support global pool which is 99% of use cases
        raise NotImplementedError(
            "AdaptiveAvgPool2d only supported for output_size=(1,1) (Global Pool)"
        )


@OpRegistry.register_function(torch.nn.functional.log_softmax)
@OpRegistry.register_module(torch.nn.LogSoftmax)
def convert_log_softmax(builder: OVGraphBuilder, node, *args, **kwargs):
    # Extract input and dim
    if isinstance(args[0], torch.nn.Module):
        inp = builder.get_input_or_constant(args[1][0])
        dim = args[0].dim if args[0].dim is not None else -1
    else:
        inp = builder.get_input_or_constant(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)

    softmax = ops.softmax(inp, axis=dim)
    return ops.log(softmax)





@OpRegistry.register_function(torch.nn.functional.linear, npu_linear_func)
def convert_linear_functional(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    bias = (
        builder.get_input_or_constant(args[2])
        if len(args) > 2
        else kwargs.get("bias", None)
    )

    mm = ops.matmul(inp, weight, transpose_a=False, transpose_b=True)
    if bias is not None:
        return ops.add(mm, bias)
    return mm


@OpRegistry.register_module(torch.nn.LayerNorm)
def convert_layernorm_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    eps = submod.eps

    # LayerNorm reduces over normalized_shape (usually the last D dims)
    # We use OpenVINO MVN or LayerNormalization
    normalized_shape = submod.normalized_shape
    rank = inp.get_output_partial_shape(0).rank.get_length()
    axes = ops.constant(list(range(rank - len(normalized_shape), rank)), dtype=np.int64)

    # Compute mean and variance
    mean = ops.reduce_mean(inp, axes, keep_dims=True)
    diff = ops.subtract(inp, mean)
    var = ops.reduce_mean(ops.multiply(diff, diff), axes, keep_dims=True)

    eps_const = ops.constant(eps, dtype=np.float32)
    std = ops.sqrt(ops.add(var, eps_const))
    norm = ops.divide(diff, std)

    res = norm
    if submod.elementwise_affine:
        w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        res = ops.add(ops.multiply(norm, w_const), b_const)

    return res

import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np
import operator
import builtins
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder
from .. import transpose as npu_transpose_func
from .. import reshape as npu_reshape_func


def _get_partial_shape(node_or_output):
    if isinstance(node_or_output, ov.Output):
        return node_or_output.get_partial_shape()
    return node_or_output.get_output_partial_shape(0)


@OpRegistry.register_function(torch.mean)
@OpRegistry.register_method("mean")
def convert_mean(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)

    if dim is None:
        rank = _get_partial_shape(inp).rank.get_length()
        axes = ops.constant(list(range(rank)), dtype=np.int64)
    elif isinstance(dim, int):
        axes = ops.constant([dim], dtype=np.int64)
    else:
        axes = ops.constant(list(dim), dtype=np.int64)

    return ops.reduce_mean(inp, axes, keep_dims=keepdim)


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
    rank = _get_partial_shape(inp).rank.get_length()
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


@OpRegistry.register_method("reshape", "view")
def convert_reshape_method(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    if len(args) > 2:
        shape = args[1:]
    else:
        shape = args[1]

    if isinstance(shape, torch.fx.Node):
        shape_node = builder.get_input_or_constant(shape)
    elif isinstance(shape, (list, tuple)):
        shape_nodes = []
        for s in shape:
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
        shape_node = ops.constant(np.array([shape], dtype=np.int64))

    return ops.reshape(inp, shape_node, special_zero=False)


@OpRegistry.register_method("expand")
def convert_expand_method(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    shape = args[1:] if len(args) > 2 else args[1]

    if isinstance(shape, torch.fx.Node):
        shape_node = builder.get_input_or_constant(shape)
    elif isinstance(shape, (list, tuple)):
        shape_nodes = []
        for s in shape:
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
        shape_node = ops.constant(np.array([shape], dtype=np.int64))

    return ops.broadcast(inp, shape_node)


@OpRegistry.register_method("transpose")
def convert_transpose_method(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim0, dim1 = args[1], args[2]

    rank = len(_get_partial_shape(inp))
    perm = list(range(rank))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

    return ops.transpose(inp, ops.constant(np.array(perm, dtype=np.int64)))


@OpRegistry.register_method("permute")
def convert_permute_method(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dims = args[1:] if len(args) > 2 else args[1]
    return ops.transpose(inp, ops.constant(np.array(dims, dtype=np.int64)))


@OpRegistry.register_method("contiguous", "type_as", "to", "float", "half")
def convert_identity_methods(builder: OVGraphBuilder, node, args, kwargs):
    return builder.get_input_or_constant(args[0])


@OpRegistry.register_method("float")
def convert_to_float(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.convert(inp, destination_type="f32")


@OpRegistry.register_method("flatten")
@OpRegistry.register_function(torch.flatten)
def convert_flatten(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    start_dim = args[1] if len(args) > 1 else 0
    end_dim = args[2] if len(args) > 2 else -1

    rank = _get_partial_shape(inp).rank.get_length()
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += rank

    shape_pattern = []
    for i in range(start_dim):
        shape_pattern.append(0)
    shape_pattern.append(-1)
    for i in range(end_dim + 1, rank):
        shape_pattern.append(0)

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

    indices_type = indices.get_element_type()
    if not (indices_type == ov.Type.i64 or indices_type == ov.Type.i32):
        indices = ops.convert(indices, destination_type="i64")

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
    rank = _get_partial_shape(inp).rank.get_length()
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
@OpRegistry.register_method("__getitem__")
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


@OpRegistry.register_function(operator.setitem)
@OpRegistry.register_method("__setitem__")
def convert_setitem(builder: OVGraphBuilder, node, args, kwargs):
    target_node_name = args[0].name
    target_ov = builder.get_input(target_node_name)
    indices_raw = args[1]
    value_ov = builder.get_input_or_constant(args[2])

    if not isinstance(indices_raw, tuple):
        indices_raw = (indices_raw,)

    all_int = all(isinstance(i, (int, torch.fx.Node)) for i in indices_raw)
    if all_int:
        coord_nodes = []
        for idx in indices_raw:
            if isinstance(idx, int):
                coord_nodes.append(ops.constant([idx], dtype=np.int64))
            else:
                v = builder.get_input(idx.name)
                v = ops.reshape(v, ops.constant([1], dtype=np.int64), False)
                coord_nodes.append(v)
        coord = ops.concat(coord_nodes, axis=0)
        coord = ops.unsqueeze(coord, ops.constant([0], dtype=np.int64))
        val_flat = ops.reshape(value_ov, ops.constant([1], dtype=np.int64), False)
        res = ops.scatter_nd_update(target_ov, coord, val_flat)
        builder.node_map[target_node_name] = res
        return res

    res = target_ov

    def _make_scalar(v):
        if isinstance(v, int):
            return ops.constant([v], dtype=np.int64)
        node_v = builder.get_input(v.name)
        return ops.reshape(node_v, ops.constant([1], dtype=np.int64), False)

    for raw_dim, idx in enumerate(indices_raw):
        actual_dim = raw_dim

        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop

            ax = ops.constant([actual_dim], dtype=np.int64)
            st = ops.constant([1], dtype=np.int64)

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


@OpRegistry.register_function(torch.squeeze)
@OpRegistry.register_method("squeeze")
def convert_squeeze(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    if len(args) > 1:
        dim = args[1]
        axes = ops.constant(np.array([dim]), dtype=np.int64)
        return ops.squeeze(inp, axes)
    elif "dim" in kwargs:
        dim = kwargs["dim"]
        axes = ops.constant(np.array([dim]), dtype=np.int64)
        return ops.squeeze(inp, axes)
    else:
        return ops.squeeze(inp)


@OpRegistry.register_function(torch.unsqueeze)
@OpRegistry.register_method("unsqueeze")
def convert_unsqueeze(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = args[1] if len(args) > 1 else kwargs["dim"]
    axes = ops.constant(np.array([dim]), dtype=np.int64)
    return ops.unsqueeze(inp, axes)


@OpRegistry.register_method("contiguous", "type_as", "to")
def convert_noop(builder: OVGraphBuilder, node, args, kwargs):
    return builder.get_input_or_constant(args[0])


@OpRegistry.register_function(builtins.getattr)
def convert_getattr(builder: OVGraphBuilder, node, args, kwargs):
    # args can be passed either as a list/tuple in args, or in node.args
    fx_args = args if args is not None else node.args
    if not fx_args:
        raise RuntimeError("getattr call has no arguments")

    obj_node = fx_args[0]
    attr_name = fx_args[1]

    if attr_name == "shape":
        if obj_node in builder.node_values:
            val = builder.node_values[obj_node]
            if hasattr(val, "shape"):
                return ops.constant(np.array(list(val.shape), dtype=np.int64))
        if hasattr(obj_node, "meta") and obj_node.meta and "tensor_meta" in obj_node.meta:
            shape = list(obj_node.meta["tensor_meta"].shape)
            return ops.constant(np.array(shape, dtype=np.int64))
        else:
            # Fallback to get_partial_shape of the openvino node if it exists
            try:
                ov_node = builder.get_input(obj_node.name)
                shape = list(_get_partial_shape(ov_node).get_shape())
                return ops.constant(np.array(shape, dtype=np.int64))
            except Exception:
                raise RuntimeError(f"Cannot get shape for {obj_node.name}: no meta or shape info")

    elif attr_name == "device":
        return ops.constant(np.array([0], dtype=np.int32))
    elif attr_name == "dtype":
        # Return a dummy int32 to represent a dtype; most ops handle types automatically in OV
        return ops.constant(np.array([0], dtype=np.int32))

    raise NotImplementedError(f"getattr({attr_name}) not implemented")

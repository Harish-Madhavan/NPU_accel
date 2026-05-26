import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np
from ..registry import OpRegistry
from ..graph_builder import OVGraphBuilder
from .. import rmsnorm as npu_rmsnorm_func
from .. import linear as npu_linear_func
from ..functional import quantized_linear as npu_quantized_linear_func


def _get_partial_shape(node_or_output):
    if isinstance(node_or_output, ov.Output):
        return node_or_output.get_partial_shape()
    return node_or_output.get_output_partial_shape(0)


@OpRegistry.register_module(torch.nn.Linear)
def convert_linear_module(builder: OVGraphBuilder, node, submod, args, kwargs):
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

    inp, w_const = builder.align_types(inp, w_const)
    mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)

    if submod.bias is not None:
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        mm, b_const = builder.align_types(mm, b_const)
        return ops.add(mm, b_const)
    return mm


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


@OpRegistry.register_function(npu_quantized_linear_func)
def convert_quantized_linear(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    scale = builder.get_input_or_constant(args[2])

    zero_point = (
        builder.get_input_or_constant(args[3])
        if len(args) > 3
        else kwargs.get("zero_point", None)
    )
    bias = (
        builder.get_input_or_constant(args[4])
        if len(args) > 4
        else kwargs.get("bias", None)
    )

    weight_tensor = builder.node_values.get(args[1], None)
    is_int4 = (
        weight_tensor is not None
        and isinstance(weight_tensor, torch.Tensor)
        and weight_tensor.dtype == torch.uint8
        and len(weight_tensor.shape) == 2
    )

    if is_int4:
        c_out, c_in_half = weight_tensor.shape
        c_in = c_in_half * 2
        inp_type = inp.get_element_type()
        w_cast = ops.convert(weight, destination_type=inp_type)

        scale_16 = ops.constant(16.0, dtype=np.float32)
        w_div = ops.divide(w_cast, scale_16)
        w_odd = ops.floor(w_div)

        w_odd_mul = ops.multiply(w_odd, scale_16)
        w_even = ops.subtract(w_cast, w_odd_mul)

        shape_3d = ops.constant([c_out, c_in_half, 1], dtype=np.int64)
        w_even_3d = ops.reshape(w_even, shape_3d, special_zero=False)
        w_odd_3d = ops.reshape(w_odd, shape_3d, special_zero=False)

        w_concat = ops.concat([w_even_3d, w_odd_3d], axis=2)

        shape_final = ops.constant([c_out, c_in], dtype=np.int64)
        weight_float = ops.reshape(w_concat, shape_final, special_zero=False)
    else:
        weight_float = ops.convert(weight, destination_type=inp.get_element_type())

    if zero_point is not None:
        zp_cast = ops.convert(zero_point, destination_type=inp.get_element_type())
        weight_float = ops.subtract(weight_float, zp_cast)

    scale_cast = ops.convert(scale, destination_type=inp.get_element_type())
    weight_float = ops.multiply(weight_float, scale_cast)

    mm = ops.matmul(inp, weight_float, transpose_a=False, transpose_b=True)

    if bias is not None:
        bias_cast = ops.convert(bias, destination_type=inp.get_element_type())
        return ops.add(mm, bias_cast)
    return mm


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


@OpRegistry.register_function(npu_rmsnorm_func)
def convert_rmsnorm(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    eps = args[2]

    x_sq = ops.multiply(inp, inp)
    rank = _get_partial_shape(inp).rank.get_length()
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
    weight = builder.get_input_or_constant(
        kwargs.get("weight", args[2] if len(args) > 2 else None)
    )
    bias = builder.get_input_or_constant(
        kwargs.get("bias", args[3] if len(args) > 3 else None)
    )
    eps = kwargs.get("eps", args[4] if len(args) > 4 else 1e-5)

    rank = _get_partial_shape(inp).rank.get_length()
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


@OpRegistry.register_module(torch.nn.LayerNorm)
def convert_layernorm_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    eps = submod.eps

    normalized_shape = submod.normalized_shape
    rank = _get_partial_shape(inp).rank.get_length()
    axes = ops.constant(list(range(rank - len(normalized_shape), rank)), dtype=np.int64)

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


@OpRegistry.register_function(torch.nn.functional.batch_norm)
def convert_batch_norm_functional(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    mean = builder.get_input_or_constant(
        kwargs.get("running_mean", args[1] if len(args) > 1 else None)
    )
    var = builder.get_input_or_constant(
        kwargs.get("running_var", args[2] if len(args) > 2 else None)
    )
    weight = builder.get_input_or_constant(
        kwargs.get("weight", args[3] if len(args) > 3 else None)
    )
    bias = builder.get_input_or_constant(
        kwargs.get("bias", args[4] if len(args) > 4 else None)
    )
    eps = kwargs.get("eps", args[7] if len(args) > 7 else 1e-5)

    if mean is None or var is None:
        raise ValueError(
            "running_mean and running_var must be provided for functional batch_norm"
        )

    axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
    mean_4d = ops.unsqueeze(mean, axes)
    var_4d = ops.unsqueeze(var, axes)

    eps_const = ops.constant(eps, dtype=np.float32)
    std = ops.sqrt(ops.add(var_4d, eps_const))
    norm = ops.divide(ops.subtract(inp, mean_4d), std)

    res = norm
    if weight is not None:
        w_4d = ops.unsqueeze(weight, axes)
        res = ops.multiply(res, w_4d)
    if bias is not None:
        b_4d = ops.unsqueeze(bias, axes)
        res = ops.add(res, b_4d)

    return res


@OpRegistry.register_module(torch.nn.BatchNorm2d)
def convert_batchnorm2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    eps = submod.eps
    axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)

    if (
        submod.track_running_stats
        and submod.running_mean is not None
        and submod.running_var is not None
    ):
        mean_const = builder.add_constant(
            f"{node.target}.running_mean", submod.running_mean
        )
        var_const = builder.add_constant(
            f"{node.target}.running_var", submod.running_var
        )
        mean_4d = ops.unsqueeze(mean_const, axes)
        var_4d = ops.unsqueeze(var_const, axes)
    else:
        reduce_axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
        mean_4d = ops.reduce_mean(inp, reduce_axes, keep_dims=True)
        diff = ops.subtract(inp, mean_4d)
        var_4d = ops.reduce_mean(
            ops.multiply(diff, diff),
            reduce_axes,
            keep_dims=True,
        )

    eps_const = ops.constant(eps, dtype=np.float32)

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


@OpRegistry.register_function(torch.nn.functional.scaled_dot_product_attention)
def convert_sdpa(builder: OVGraphBuilder, node, args, kwargs):
    query = builder.get_input_or_constant(args[0])
    key = builder.get_input_or_constant(args[1])
    value = builder.get_input_or_constant(args[2])

    attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
    scale = kwargs.get("scale", args[6] if len(args) > 6 else None)

    ov_mask = None
    if attn_mask is not None:
        ov_mask = builder.get_input_or_constant(attn_mask)

    if scale is None:
        q_shape = _get_partial_shape(query)
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


def convert_stateful_kv_cache(builder: OVGraphBuilder, node, submod, args, kwargs):
    new_kv = builder.get_input_or_constant(args[0])

    ov_type = ov.Type.f32
    if submod.dtype == torch.float16:
        ov_type = ov.Type.f16

    cache_shape = [submod.batch_size, submod.max_seq_len, submod.num_heads, submod.head_dim]

    info_cache = ov.op.util.VariableInfo()
    info_cache.data_shape = ov.PartialShape(cache_shape)
    info_cache.data_type = ov_type
    info_cache.variable_id = f"{node.name}_cache_state"

    cache_var = ov.op.util.Variable(info_cache)
    builder.variables.append(cache_var)

    read_cache = ops.read_value(cache_var)

    info_pos = ov.op.util.VariableInfo()
    info_pos.data_shape = ov.PartialShape([])
    info_pos.data_type = ov.Type.f32
    info_pos.variable_id = f"{node.name}_pos_state"

    pos_var = ov.op.util.Variable(info_pos)
    builder.variables.append(pos_var)

    read_pos = ops.read_value(pos_var)

    new_kv_shape = ops.shape_of(new_kv)
    seq_len = ops.gather(new_kv_shape, ops.constant([1], dtype=np.int32), ops.constant([0], dtype=np.int32))
    seq_len = ops.squeeze(seq_len, ops.constant([0], dtype=np.int32))
    seq_len_i32 = ops.convert(seq_len, destination_type="i32")

    read_pos_i32 = ops.convert(read_pos, destination_type="i32")
    indices = ops.range(read_pos_i32, ops.add(read_pos_i32, seq_len_i32), ops.constant(1, dtype=np.int32), output_type="i32")

    updated_cache = ops.scatter_update(read_cache, indices, new_kv, ops.constant([1], dtype=np.int32))

    assign_cache = ops.assign(updated_cache, cache_var)
    builder.sinks.append(assign_cache)

    new_pos = ops.add(read_pos, ops.convert(seq_len_i32, destination_type="f32"))
    assign_pos = ops.assign(new_pos, pos_var)
    builder.sinks.append(assign_pos)

    return updated_cache


from .._functional import update_kv_cache as npu_update_kv_cache_func

@OpRegistry.register_function(npu_update_kv_cache_func)
def convert_update_kv_cache(builder: OVGraphBuilder, node, args, kwargs):
    cache = builder.get_input_or_constant(args[0])
    new_kv = builder.get_input_or_constant(args[1])
    position = builder.get_input_or_constant(args[2])

    cache, new_kv = builder.align_types(cache, new_kv)

    if not isinstance(position, ov.Node):
        position = ops.constant([position], dtype=np.int64)
    else:
        position = ops.reshape(position, ops.constant([1], dtype=np.int64), special_zero=False)

    pos_type = position.get_element_type()
    if not (pos_type == ov.Type.i64 or pos_type == ov.Type.i32):
        position = ops.convert(position, destination_type="i64")

    new_kv_shape = ops.shape_of(new_kv)
    seq_len = ops.gather(new_kv_shape, ops.constant([1], dtype=np.int32), ops.constant([0], dtype=np.int32))
    seq_len = ops.squeeze(seq_len, ops.constant([0], dtype=np.int32))
    seq_len_i32 = ops.convert(seq_len, destination_type="i32")

    pos_i32 = ops.convert(position, destination_type="i32")
    indices = ops.range(pos_i32, ops.add(pos_i32, seq_len_i32), ops.constant(1, dtype=np.int32), output_type="i32")

    return ops.scatter_update(cache, indices, new_kv, ops.constant([1], dtype=np.int32))


# Break circular dependency dynamically
try:
    from ..functional import NPUStatefulKVCache
    OpRegistry.register_module(NPUStatefulKVCache)(convert_stateful_kv_cache)
except ImportError:
    pass

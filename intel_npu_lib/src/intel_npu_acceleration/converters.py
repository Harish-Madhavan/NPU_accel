import torch
import torch.nn as nn
import openvino.opset13 as ops
import numpy as np
import operator
import builtins
from typing import Any
from .registry import OpRegistry
from .graph_builder import OVGraphBuilder
from . import transpose as npu_transpose_func
from . import reshape as npu_reshape_func
from . import rmsnorm as npu_rmsnorm_func

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
    
    attn_mask = kwargs.get('attn_mask', args[3] if len(args) > 3 else None)
    dropout_p = kwargs.get('dropout_p', args[4] if len(args) > 4 else 0.0)
    is_causal = kwargs.get('is_causal', args[5] if len(args) > 5 else False)
    
    # Scale: default is 1 / sqrt(head_dim)
    # OpenVINO SDPA might compute this automatically if scale is not provided?
    # No, usually need to provide it or it defaults.
    # Torch doc: "If scale is None, it defaults to 1 / sqrt(query.size(-1))"
    # We might need to construct the scale constant.
    
    # Check if we need to provide scale.
    # OpenVINO ops.scaled_dot_product_attention(query, key, value, attention_mask=None, scale=None, causal=False)
    
    ov_mask = None
    if attn_mask is not None:
        ov_mask = builder.get_input_or_constant(attn_mask)
        
    # is_causal handling
    # If is_causal is True, OpenVINO handles it.
    
    return ops.scaled_dot_product_attention(query, key, value, attention_mask=ov_mask, causal=is_causal)

@OpRegistry.register_function(torch.add, operator.add)
def convert_add(builder: OVGraphBuilder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
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
    
    t0 = inp0.get_element_type()
    t1 = inp1.get_element_type()
    target_type = np.float32
    if '16' in str(t0) and '16' in str(t1): target_type = np.float16
    elif '16' in str(t0) and not 'f' in str(t1): target_type = np.float16
    elif '16' in str(t1) and not 'f' in str(t0): target_type = np.float16
    
    if t0 != target_type: inp0 = ops.convert(inp0, destination_type=target_type)
    if t1 != target_type: inp1 = ops.convert(inp1, destination_type=target_type)
    
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
    
    t0 = inp0.get_element_type()
    t1 = inp1.get_element_type()
    if str(t0) in ['i8', 'u8'] or str(t1) in ['i8', 'u8']:
         target_type = np.int32
         if str(t0) != 'i32': inp0 = ops.convert(inp0, destination_type=target_type)
         if str(t1) != 'i32': inp1 = ops.convert(inp1, destination_type=target_type)

    return ops.matmul(inp0, inp1, transpose_a=False, transpose_b=False)

@OpRegistry.register_function(torch.relu, torch.nn.functional.relu)
def convert_relu(builder: OVGraphBuilder, node, args, kwargs):
    return ops.relu(builder.get_input_or_constant(args[0]))

@OpRegistry.register_function(torch.nn.functional.gelu)
def convert_gelu(builder: OVGraphBuilder, node, args, kwargs):
    return ops.gelu(builder.get_input_or_constant(args[0]), approximation_mode="erf")

@OpRegistry.register_function(torch.nn.functional.silu)
def convert_silu(builder: OVGraphBuilder, node, args, kwargs):
    return ops.swish(builder.get_input_or_constant(args[0]))

@OpRegistry.register_function(torch.rsqrt)
def convert_rsqrt(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    exp_node = ops.constant(-0.5, dtype=np.float32)
    return ops.power(inp, exp_node)

@OpRegistry.register_function(torch.mean)
@OpRegistry.register_method("mean")
def convert_mean(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = node.kwargs.get('dim', args[1] if len(args) > 1 else None)
    keepdim = node.kwargs.get('keepdim', args[2] if len(args) > 2 else False)
    
    if dim is None:
        raise RuntimeError("Mean without dim not supported yet")
    
    if isinstance(dim, int):
        axes = ops.constant([dim], dtype=np.int64)
    else:
        axes = ops.constant(list(dim), dtype=np.int64)
        
    return ops.reduce_mean(inp, axes, keep_dims=keepdim)

@OpRegistry.register_function(torch.softmax, torch.nn.functional.softmax)
def convert_softmax(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = node.kwargs.get('dim', args[1] if len(args) > 1 else -1)
    return ops.softmax(inp, axis=dim)

@OpRegistry.register_function(torch.cat)
def convert_cat(builder: OVGraphBuilder, node, args, kwargs):
    tensors_list_node = node.args[0]
    dim = node.kwargs['dim']
    # Map input nodes
    ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
    return ops.concat(ov_inputs, axis=dim)

@OpRegistry.register_function(torch.stack)
def convert_stack(builder: OVGraphBuilder, node, args, kwargs):
    tensors_list_node = node.args[0]
    dim = node.kwargs['dim']
    ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
    
    unsqueezed_inputs = []
    for ov_input in ov_inputs:
        unsqueezed_inputs.append(ops.unsqueeze(ov_input, ops.constant(np.array([dim]), dtype=np.int64)))
    
    return ops.concat(unsqueezed_inputs, axis=dim)

@OpRegistry.register_function(torch.transpose, npu_transpose_func)
@OpRegistry.register_method("transpose")
def convert_transpose(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim0 = args[1]
    dim1 = args[2]
    rank = inp.get_output_partial_shape(0).rank.get_length()
    perm = list(range(rank))
    if dim0 < 0: dim0 += rank
    if dim1 < 0: dim1 += rank
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
            shape_nodes.append(ops.reshape(val_node, ops.constant([1], dtype=np.int64), special_zero=False))
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

@OpRegistry.register_function(builtins.getattr)
def convert_getattr(builder: OVGraphBuilder, node, args, kwargs):
    obj_node = args[0]
    attr_name = args[1]
    if attr_name == 'shape':
        if 'tensor_meta' in obj_node.meta:
            shape = list(obj_node.meta['tensor_meta'].shape)
            return ops.constant(np.array(shape, dtype=np.int64))
        else:
            raise RuntimeError(f"Cannot get shape for {obj_node.name}: no meta info")
    elif attr_name == 'device':
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
                 shape_nodes.append(ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False))
         shape_node = ops.concat(shape_nodes, axis=0)
    else:
         shape_node = builder.get_input_or_constant(size)

    val_node = builder.get_input_or_constant(fill_value)
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
    target_node_name = args[0].name
    target_ov = builder.get_input(target_node_name)
    indices = args[1]
    value_ov = builder.get_input_or_constant(args[2])
    
    slice_dim = -1
    slice_start = None
    slice_end = None
    
    if isinstance(indices, tuple):
        for dim, idx in enumerate(indices):
            if isinstance(idx, slice):
                if idx.start is not None and idx.stop is not None:
                     if slice_dim != -1:
                         pass
                     slice_dim = dim
                     slice_start = idx.start
                     slice_end = idx.stop
            elif isinstance(idx, (int, torch.fx.Node)):
                 pass
    
    if slice_dim == -1:
         if isinstance(indices, slice):
              slice_dim = 0
              slice_start = indices.start
              slice_end = indices.stop
         else:
              raise NotImplementedError("Could not find slice in setitem")

    def make_slice(inp, axis, start_val, end_val):
         s = builder.get_input_or_constant(start_val)
         e = builder.get_input_or_constant(end_val)
         
         if isinstance(start_val, int): s = ops.constant([start_val], dtype=np.int64)
         if isinstance(end_val, int): e = ops.constant([end_val], dtype=np.int64)
         
         if s.get_output_partial_shape(0).rank.get_length() == 0:
              s = ops.reshape(s, ops.constant([1], dtype=np.int64), False)
         if e.get_output_partial_shape(0).rank.get_length() == 0:
              e = ops.reshape(e, ops.constant([1], dtype=np.int64), False)

         ax = ops.constant([axis], dtype=np.int64)
         st = ops.constant([1], dtype=np.int64) 
         
         return ops.slice(inp, s, e, st, ax)

    p1 = make_slice(target_ov, slice_dim, 0, slice_start)
    
    shape_node = ops.shape_of(target_ov)
    dim_size = ops.gather(shape_node, ops.constant([slice_dim], dtype=np.int64), ops.constant([0], dtype=np.int64))
    dim_size = ops.reshape(dim_size, ops.constant([1], dtype=np.int64), False)
    
    p2 = make_slice(target_ov, slice_dim, slice_end, dim_size)
    
    res = ops.concat([p1, value_ov, p2], axis=slice_dim)
    
    builder.node_map[target_node_name] = res
    return res

@OpRegistry.register_function(torch.where)
def convert_where(builder: OVGraphBuilder, node, args, kwargs):
    condition = builder.get_input_or_constant(args[0])
    x = builder.get_input_or_constant(args[1])
    y = builder.get_input_or_constant(args[2])
    x, y = builder.align_types(x, y)
    return ops.select(condition, x, y)

@OpRegistry.register_function(torch.index_select)
def convert_index_select(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = args[1]
    index = builder.get_input_or_constant(args[2])
    
    axis = ops.constant(np.array([dim]), dtype=np.int64)
    return ops.gather(inp, index, axis)

@OpRegistry.register_function(torch.triu)
def convert_triu(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    diagonal = node.kwargs.get('diagonal', args[1] if len(args) > 1 else 0)
    
    shape = ops.shape_of(inp)
    rank = inp.get_output_partial_shape(0).rank.get_length()
    if rank < 2:
        return inp 
        
    shape_i32 = ops.convert(shape, destination_type=np.int32)
    h_idx = ops.gather(shape_i32, ops.constant([rank-2], dtype=np.int32), ops.constant([0], dtype=np.int32))
    w_idx = ops.gather(shape_i32, ops.constant([rank-1], dtype=np.int32), ops.constant([0], dtype=np.int32))
    
    h_idx = ops.squeeze(h_idx, ops.constant([0], dtype=np.int32))
    w_idx = ops.squeeze(w_idx, ops.constant([0], dtype=np.int32))
    
    row_range = ops.range(ops.constant(0, dtype=np.int32), h_idx, ops.constant(1, dtype=np.int32), output_type="i32")
    row_range = ops.unsqueeze(row_range, ops.constant([1], dtype=np.int32))
    
    col_range = ops.range(ops.constant(0, dtype=np.int32), w_idx, ops.constant(1, dtype=np.int32), output_type="i32")
    col_range = ops.unsqueeze(col_range, ops.constant([0], dtype=np.int32))
    
    if isinstance(diagonal, (int, float)):
         diag_const = ops.constant(diagonal, dtype=np.int32)
    else:
         diag_const = builder.get_input_or_constant(diagonal)
         if '32' not in str(diag_const.get_element_type()):
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
        if isinstance(idx, slice): idx = (idx,)
        
        begin_nodes, end_nodes, strides_nodes = [], [], []
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask = [], [], [], [], []
        
        for item in idx:
            if item is Ellipsis:
                begin_nodes.append(ops.constant([0], dtype=np.int64))
                end_nodes.append(ops.constant([0], dtype=np.int64))
                strides_nodes.append(ops.constant([1], dtype=np.int64))
                begin_mask.append(0); end_mask.append(0); ellipsis_mask.append(1)
                new_axis_mask.append(0); shrink_axis_mask.append(0)
                continue
            
            if isinstance(item, slice):
                if item.start is None:
                    start_val = ops.constant([0], dtype=np.int64)
                    begin_mask.append(1)
                else:
                    start_val = ops.reshape(builder.get_input_or_constant(item.start), ops.constant([1], dtype=np.int64), special_zero=False)
                    begin_mask.append(0)
                
                if item.stop is None:
                    stop_val = ops.constant([0], dtype=np.int64)
                    end_mask.append(1)
                else:
                    stop_val = ops.reshape(builder.get_input_or_constant(item.stop), ops.constant([1], dtype=np.int64), special_zero=False)
                    end_mask.append(0)
                    
                if item.step is None:
                    step_val = ops.constant([1], dtype=np.int64)
                else:
                    step_val = ops.reshape(builder.get_input_or_constant(item.step), ops.constant([1], dtype=np.int64), special_zero=False)
                
                begin_nodes.append(start_val)
                end_nodes.append(stop_val)
                strides_nodes.append(step_val)
                ellipsis_mask.append(0); new_axis_mask.append(0); shrink_axis_mask.append(0)

            elif isinstance(item, int):
                 start_val = ops.constant([item], dtype=np.int64)
                 stop_val = ops.constant([item + 1], dtype=np.int64)
                 step_val = ops.constant([1], dtype=np.int64)
                 begin_nodes.append(start_val); end_nodes.append(stop_val); strides_nodes.append(step_val)
                 begin_mask.append(0); end_mask.append(0); ellipsis_mask.append(0)
                 new_axis_mask.append(0); shrink_axis_mask.append(1)
            
            elif item is None:
                 begin_nodes.append(ops.constant([0], dtype=np.int64))
                 end_nodes.append(ops.constant([0], dtype=np.int64))
                 strides_nodes.append(ops.constant([1], dtype=np.int64))
                 begin_mask.append(0); end_mask.append(0); ellipsis_mask.append(0)
                 new_axis_mask.append(1); shrink_axis_mask.append(0)

            else:
                 v = builder.get_input(item.name)
                 start_val = ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False)
                 one = ops.constant([1], dtype=np.int64)
                 stop_val = ops.add(start_val, one)
                 step_val = ops.constant([1], dtype=np.int64)
                 begin_nodes.append(start_val); end_nodes.append(stop_val); strides_nodes.append(step_val)
                 begin_mask.append(0); end_mask.append(0); ellipsis_mask.append(0)
                 new_axis_mask.append(0); shrink_axis_mask.append(1)

        begin_t = ops.concat(begin_nodes, axis=0)
        end_t = ops.concat(end_nodes, axis=0)
        strides_t = ops.concat(strides_nodes, axis=0)
        
        return ops.strided_slice(inp, begin_t, end_t, strides_t, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask)

    elif isinstance(idx, torch.fx.Node):
        idx_node = builder.get_input(idx.name)
        axis = ops.constant([0], dtype=np.int64)
        return ops.gather(inp, idx_node, axis)

    raise NotImplementedError(f"getitem index {type(idx)} not implemented")

# --- Module Converters ---

@OpRegistry.register_module(torch.nn.Linear)
def convert_linear_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    
    mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)
    res = mm
    if submod.bias is not None:
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        res = ops.add(mm, b_const)
    return res

@OpRegistry.register_module(torch.nn.Embedding)
def convert_embedding_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    indices = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    axis = ops.constant([0], dtype=np.int64)
    return ops.gather(w_const, indices, axis)

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
    return ops.convert(inp, destination_type='f32')

@OpRegistry.register_method("flatten")
def convert_flatten(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    start_dim = args[1] if len(args) > 1 else 0
    end_dim = args[2] if len(args) > 2 else -1
    
    rank = inp.get_output_partial_shape(0).rank.get_length()
    if start_dim < 0: start_dim += rank
    if end_dim < 0: end_dim += rank
    
    shape_pattern = []
    for i in range(start_dim):
        shape_pattern.append(0) # Copy
    shape_pattern.append(-1) # Merge
    for i in range(end_dim + 1, rank):
        shape_pattern.append(0) # Copy
        
    return ops.reshape(inp, shape_pattern, special_zero=True)

@OpRegistry.register_method("size")
def convert_size(builder: OVGraphBuilder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    if len(args) > 1:
        dim = args[1]
        shape = ops.shape_of(inp)
        dim_val = ops.gather(shape, ops.constant([dim], dtype=np.int64), ops.constant([0], dtype=np.int64))
        return ops.squeeze(dim_val, ops.constant([0], dtype=np.int64))
    else:
        return ops.shape_of(inp)

@OpRegistry.register_function(torch.nn.functional.conv2d)
def convert_conv2d(builder: OVGraphBuilder, node, args, kwargs):
    # args: input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    inp = builder.get_input_or_constant(args[0])
    weight = builder.get_input_or_constant(args[1])
    bias = builder.get_input_or_constant(args[2]) if len(args) > 2 else kwargs.get('bias', None)
    
    stride = kwargs.get('stride', args[3] if len(args) > 3 else 1)
    padding = kwargs.get('padding', args[4] if len(args) > 4 else 0)
    dilation = kwargs.get('dilation', args[5] if len(args) > 5 else 1)
    groups = kwargs.get('groups', args[6] if len(args) > 6 else 1)
    
    stride_list = _to_list(stride)
    padding_list = _to_list(padding)
    dilation_list = _to_list(dilation)
    
    # Create OV Strides/Pads
    ov_strides = np.array(stride_list, dtype=np.int64)
    ov_pads_begin = np.array(padding_list, dtype=np.int64)
    ov_pads_end = np.array(padding_list, dtype=np.int64)
    ov_dilations = np.array(dilation_list, dtype=np.int64)
    
    if groups == 1:
        conv = ops.convolution(inp, weight, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations)
    else:
        raise NotImplementedError("Groups > 1 not fully supported in compiler yet.")
        
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
    kernel_size = kwargs.get('kernel_size', args[1])
    stride = kwargs.get('stride', args[2] if len(args) > 2 else None)
    padding = kwargs.get('padding', args[3] if len(args) > 3 else 0)
    dilation = kwargs.get('dilation', args[4] if len(args) > 4 else 1)
    ceil_mode = kwargs.get('ceil_mode', args[5] if len(args) > 5 else False)
    
    if stride is None: stride = kernel_size
    
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
        rounding_type=rounding_type
    )

@OpRegistry.register_module(torch.nn.Conv2d)
def convert_conv2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    
    stride_list = _to_list(submod.stride)
    padding_list = _to_list(submod.padding)
    dilation_list = _to_list(submod.dilation)
    
    ov_strides = np.array(stride_list, dtype=np.int64)
    ov_pads_begin = np.array(padding_list, dtype=np.int64)
    ov_pads_end = np.array(padding_list, dtype=np.int64)
    ov_dilations = np.array(dilation_list, dtype=np.int64)
    
    # Groups = 1 only for now
    conv = ops.convolution(inp, w_const, ov_strides, ov_pads_begin, ov_pads_end, ov_dilations)
    
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
        rounding_type=rounding_type
    )

@OpRegistry.register_module(torch.nn.BatchNorm2d)
def convert_batchnorm2d_module(builder: OVGraphBuilder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    
    # We need to broadcast statistics to (1, C, 1, 1)
    axes = ops.constant(np.array([0, 2, 3]), dtype=np.int64)
    
    mean = submod.running_mean
    var = submod.running_var
    eps = submod.eps
    
    if mean is None or var is None:
        raise NotImplementedError("BatchNorm2d without tracking stats not supported in inference compiler.")

    mean_const = builder.add_constant(f"{node.target}.running_mean", mean)
    var_const = builder.add_constant(f"{node.target}.running_var", var)
    eps_const = ops.constant(eps, dtype=np.float32)
    
    mean_4d = ops.unsqueeze(mean_const, axes)
    var_4d = ops.unsqueeze(var_const, axes)
    
    # x - mean
    sub = ops.subtract(inp, mean_4d)
    
    # sqrt(var + eps)
    std = ops.sqrt(ops.add(var_4d, eps_const))
    
    # div
    norm = ops.divide(sub, std)
    
    res = norm
    
    if submod.affine:
        weight = submod.weight
        bias = submod.bias
        w_const = builder.add_constant(f"{node.target}.weight", weight)
        b_const = builder.add_constant(f"{node.target}.bias", bias)
        
        w_4d = ops.unsqueeze(w_const, axes)
        b_4d = ops.unsqueeze(b_const, axes)
        
        # * weight + bias
        res = ops.multiply(res, w_4d)
        res = ops.add(res, b_4d)
        
    return res

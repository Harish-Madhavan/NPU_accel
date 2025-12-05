
import torch
import torch.nn as nn
import torch.fx
from torch.fx.interpreter import Interpreter
import openvino as ov
import openvino.runtime.opset13 as ops
import numpy as np
from typing import Dict, Any, List, Union, Optional
import operator
import builtins
from . import transpose as npu_transpose_func
from . import reshape as npu_reshape_func
from . import rmsnorm as npu_rmsnorm_func

# --- Registry ---

class OpRegistry:
    _function_converters = {}
    _method_converters = {}
    _module_converters = {}

    @classmethod
    def register_function(cls, *targets):
        def decorator(func):
            for t in targets:
                cls._function_converters[t] = func
            return func
        return decorator

    @classmethod
    def register_method(cls, *names):
        def decorator(func):
            for n in names:
                cls._method_converters[n] = func
            return func
        return decorator

    @classmethod
    def register_module(cls, *types):
        def decorator(func):
            for t in types:
                cls._module_converters[t] = func
            return func
        return decorator
    
    @classmethod
    def get_function(cls, target):
        return cls._function_converters.get(target)

    @classmethod
    def get_method(cls, name):
        return cls._method_converters.get(name)
    
    @classmethod
    def get_module(cls, module_type):
        # Handle subclasses if needed, but direct match for now
        return cls._module_converters.get(module_type)

# --- Interpreter ---

class ValueCapturingInterpreter(Interpreter):
    def __init__(self, module, garbage_collect_values=False):
        super().__init__(module, garbage_collect_values)
        self.node_values = {}

    def run_node(self, n):
        val = super().run_node(n)
        self.node_values[n] = val
        return val

# --- Graph Builder & Helper ---

class OVGraphBuilder:
    def __init__(self, node_values: Dict[torch.fx.Node, Any] = None):
        self.node_map: Dict[str, Any] = {} # Maps fx node name to OV node output
        self.parameters: List[Any] = []
        self.result_nodes: List[Any] = [] # List of OV nodes that are outputs
        self.node_values = node_values if node_values is not None else {}
        
    def get_input(self, node_name: str):
        if node_name not in self.node_map:
            raise RuntimeError(f"Node {node_name} not found in graph map.")
        return self.node_map[node_name]

    def get_input_or_constant(self, arg: Any):
        if isinstance(arg, torch.fx.Node):
            # Try to constant fold scalars (ints/floats) to support static shape requirements
            if arg in self.node_values:
                val = self.node_values[arg]
                if isinstance(val, (int, float, bool, np.integer, np.floating)):
                    # Check type and return constant
                    dtype = np.float32
                    if isinstance(val, (int, np.integer)): dtype = np.int64
                    elif isinstance(val, bool): dtype = bool
                    return ops.constant(val, dtype=dtype)

            return self.get_input(arg.name)
        elif isinstance(arg, float):
            return ops.constant(arg, dtype=np.float32)
        elif isinstance(arg, int):
            return ops.constant(arg, dtype=np.int64)
        elif isinstance(arg, (list, tuple)):
            # Recursively handle lists/tuples (e.g. for reshape shapes)
            # This might be too aggressive for complex args, but works for shapes.
            # If it's a list of nodes/ints, we might want to stack or return list.
            # For now, return list of processed items.
            return [self.get_input_or_constant(x) for x in arg]
        elif arg is None:
            return None 
        else:
            # Fallback/Error
            raise NotImplementedError(f"Cannot convert argument type {type(arg)} to OV input")

    def add_parameter(self, name: str, shape: List[int], dtype):
        # Convert torch dtype to numpy dtype
        if dtype == torch.float32: np_type = np.float32
        elif dtype == torch.float16: np_type = np.float16
        elif dtype == torch.int64: np_type = np.int64
        elif dtype == torch.int8: np_type = np.int32 
        elif dtype == torch.uint8: np_type = np.int32
        else: np_type = np.float32 # Default
        
        param = ops.parameter(ov.Shape(shape), dtype=np_type, name=name)
        self.node_map[name] = param
        self.parameters.append(param)
        return param

    def add_constant(self, name: str, tensor: torch.Tensor):
        # Tensor to numpy
        data = tensor.detach().cpu().numpy()
        try:
            const_node = ops.constant(data, name=name)
        except Exception as e:
            print(f"DEBUG: Failed to create constant for {name}. Shape: {data.shape}, Dtype: {data.dtype}")
            raise e
        self.node_map[name] = const_node
        return const_node

    def register_output(self, name: str, ov_node):
        self.node_map[name] = ov_node
        
    def align_types(self, inp0, inp1):
        t0 = inp0.get_element_type()
        t1 = inp1.get_element_type()
        
        if t0 == t1:
            return inp0, inp1
            
        is_f0 = 'f' in str(t0)
        is_f1 = 'f' in str(t1)
        
        target_type = None
        if is_f0 and is_f1:
             if '32' in str(t0) or '32' in str(t1): target_type = np.float32
             else: target_type = np.float16
        elif is_f0:
             target_type = np.float32 if '32' in str(t0) else np.float16
        elif is_f1:
             target_type = np.float32 if '32' in str(t1) else np.float16
        else:
             target_type = np.int64
             
        if target_type:
            if t0 != target_type: inp0 = ops.convert(inp0, destination_type=target_type)
            if t1 != target_type: inp1 = ops.convert(inp1, destination_type=target_type)
            
        return inp0, inp1

# --- Converters ---

@OpRegistry.register_function(torch.add, operator.add)
def convert_add(builder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    return ops.add(inp0, inp1)

@OpRegistry.register_function(torch.sub, operator.sub)
def convert_sub(builder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    return ops.subtract(inp0, inp1)

@OpRegistry.register_function(torch.mul, operator.mul)
def convert_mul(builder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    return ops.multiply(inp0, inp1)

@OpRegistry.register_function(torch.div, operator.truediv)
def convert_div(builder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    
    # Ensure float output for true div
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
def convert_floordiv(builder, node, args, kwargs):
    inp0 = builder.get_input_or_constant(args[0])
    inp1 = builder.get_input_or_constant(args[1])
    inp0, inp1 = builder.align_types(inp0, inp1)
    div = ops.divide(inp0, inp1)
    res = ops.floor(div)
    return ops.convert(res, destination_type=np.int64)

@OpRegistry.register_function(torch.pow, operator.pow)
@OpRegistry.register_method("pow")
def convert_pow(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    exponent = builder.get_input_or_constant(args[1])
    inp, exponent = builder.align_types(inp, exponent)
    return ops.power(inp, exponent)

@OpRegistry.register_function(torch.neg, operator.neg)
def convert_neg(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.negative(inp)

@OpRegistry.register_function(torch.sin)
def convert_sin(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.sin(inp)

@OpRegistry.register_function(torch.cos)
def convert_cos(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.cos(inp)

@OpRegistry.register_function(torch.matmul, torch.mm)
def convert_matmul(builder, node, args, kwargs):
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
def convert_relu(builder, node, args, kwargs):
    return ops.relu(builder.get_input_or_constant(args[0]))

@OpRegistry.register_function(torch.nn.functional.gelu)
def convert_gelu(builder, node, args, kwargs):
    return ops.gelu(builder.get_input_or_constant(args[0]), approximation_mode="erf")

@OpRegistry.register_function(torch.nn.functional.silu)
def convert_silu(builder, node, args, kwargs):
    return ops.swish(builder.get_input_or_constant(args[0]))

@OpRegistry.register_function(torch.rsqrt)
def convert_rsqrt(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    exp_node = ops.constant(-0.5, dtype=np.float32)
    return ops.power(inp, exp_node)

@OpRegistry.register_function(torch.mean)
@OpRegistry.register_method("mean")
def convert_mean(builder, node, args, kwargs):
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
def convert_softmax(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    dim = node.kwargs.get('dim', args[1] if len(args) > 1 else -1)
    return ops.softmax(inp, axis=dim)

@OpRegistry.register_function(torch.cat)
def convert_cat(builder, node, args, kwargs):
    tensors_list_node = node.args[0]
    dim = node.kwargs['dim']
    # Map input nodes
    ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
    return ops.concat(ov_inputs, axis=dim)

@OpRegistry.register_function(torch.stack)
def convert_stack(builder, node, args, kwargs):
    tensors_list_node = node.args[0]
    dim = node.kwargs['dim']
    ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
    
    unsqueezed_inputs = []
    for ov_input in ov_inputs:
        unsqueezed_inputs.append(ops.unsqueeze(ov_input, ops.constant(np.array([dim]), dtype=np.int64)))
    
    return ops.concat(unsqueezed_inputs, axis=dim)

@OpRegistry.register_function(torch.transpose, npu_transpose_func)
@OpRegistry.register_method("transpose")
def convert_transpose(builder, node, args, kwargs):
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
def convert_reshape(builder, node, args, kwargs):
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
def convert_rmsnorm(builder, node, args, kwargs):
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
def convert_getattr(builder, node, args, kwargs):
    obj_node = args[0]
    attr_name = args[1]
    if attr_name == 'shape':
        if 'tensor_meta' in obj_node.meta:
            shape = list(obj_node.meta['tensor_meta'].shape)
            return ops.constant(np.array(shape, dtype=np.int64))
        else:
            raise RuntimeError(f"Cannot get shape for {obj_node.name}: no meta info")
    elif attr_name == 'device':
        # Return a dummy constant for device, consumers should ignore it
        return ops.constant(np.array([0], dtype=np.int32))
    raise NotImplementedError(f"getattr({attr_name}) not implemented")

@OpRegistry.register_function(torch.full)
def convert_full(builder, node, args, kwargs):
    # args: size, fill_value
    size = args[0]
    fill_value = args[1]
    
    # Handle size (can be tuple, list, or node)
    if isinstance(size, (tuple, list)):
         # If items are nodes/ints
         shape_nodes = []
         for s in size:
             if isinstance(s, int):
                 shape_nodes.append(ops.constant([s], dtype=np.int64))
             else:
                 v = builder.get_input_or_constant(s)
                 shape_nodes.append(ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False))
         shape_node = ops.concat(shape_nodes, axis=0)
    else:
         # size is a node (e.g. shape tensor)
         shape_node = builder.get_input_or_constant(size)

    # Fill value
    val_node = builder.get_input_or_constant(fill_value)
    
    # Broadcast
    return ops.broadcast(val_node, shape_node)

@OpRegistry.register_function(torch.arange)
def convert_arange(builder, node, args, kwargs):
    # start, end, step
    # args can be (end,) or (start, end) or (start, end, step)
    if len(args) == 1:
        start = ops.constant(0, dtype=np.float32) # or int
        stop = builder.get_input_or_constant(args[0])
        step = ops.constant(1, dtype=np.float32)
    elif len(args) == 2:
        start = builder.get_input_or_constant(args[0])
        stop = builder.get_input_or_constant(args[1])
        step = ops.constant(1, dtype=start.get_element_type()) # match type
    else:
        start = builder.get_input_or_constant(args[0])
        stop = builder.get_input_or_constant(args[1])
        step = builder.get_input_or_constant(args[2])
        
    # Align types
    start, stop = builder.align_types(start, stop)
    start, step = builder.align_types(start, step)
    
    # OpenVINO Range: start, stop, step
    # Output type defaults to start type
    return ops.range(start, stop, step, output_type=start.get_element_type())

@OpRegistry.register_function(torch.triu)
def convert_triu(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    diagonal = node.kwargs.get('diagonal', args[1] if len(args) > 1 else 0)
    
    # SelectUpper logic
    # OV doesn't have a direct Triu op in opset1. 
    # But it has Select or we can build a mask.
    # Creating a mask of indices?
    # (i, j). i <= j - diagonal  -> keep?
    # Triu: keep if j >= i + diagonal.
    # i: row index, j: col index.
    
    shape = ops.shape_of(inp)
    # Assuming 2D or last 2 dims? 
    # PyTorch triu works on last 2 dims.
    
    rank = inp.get_output_partial_shape(0).rank.get_length()
    if rank < 2:
        return inp # No-op for 1D? PyTorch docs say: "The other dimensions are treated as batch dimensions"
        
    # Get H, W
    # We need to generate a grid of indices.
    # This is complex to implement fully generic in OV using just opset1 primitives without Loop.
    # However, usually shapes are small or fixed?
    # If dynamic, we need Range + Reshape + Broadcast.
    
    # Implementation:
    # 1. Get H, W
    shape_i32 = ops.convert(shape, destination_type=np.int32)
    h_idx = ops.gather(shape_i32, ops.constant([rank-2], dtype=np.int32), ops.constant([0], dtype=np.int32))
    w_idx = ops.gather(shape_i32, ops.constant([rank-1], dtype=np.int32), ops.constant([0], dtype=np.int32))
    
    # Squeeze to scalars for Range
    h_idx = ops.squeeze(h_idx, ops.constant([0], dtype=np.int32))
    w_idx = ops.squeeze(w_idx, ops.constant([0], dtype=np.int32))
    
    # 2. Create row indices (0..H-1) -> (H, 1)
    row_range = ops.range(ops.constant(0, dtype=np.int32), h_idx, ops.constant(1, dtype=np.int32), output_type="i32")
    row_range = ops.unsqueeze(row_range, ops.constant([1], dtype=np.int32))
    
    # 3. Create col indices (0..W-1) -> (1, W)
    col_range = ops.range(ops.constant(0, dtype=np.int32), w_idx, ops.constant(1, dtype=np.int32), output_type="i32")
    col_range = ops.unsqueeze(col_range, ops.constant([0], dtype=np.int32))
    
    # 4. Compare: j >= i + diagonal
    # col_range >= row_range + diagonal
    
    if isinstance(diagonal, (int, float)):
         diag_const = ops.constant(diagonal, dtype=np.int32)
    else:
         diag_const = builder.get_input_or_constant(diagonal)
         # Ensure i32
         if '32' not in str(diag_const.get_element_type()): # simplified check
              diag_const = ops.convert(diag_const, destination_type=np.int32)
              
    threshold = ops.add(row_range, diag_const)
    mask = ops.greater_equal(col_range, threshold) # Broadcasts to (H, W)
    
    # 5. Select
    # If true, keep inp, else 0
    zero = ops.constant(0, dtype=inp.get_element_type())
    
    # Mask needs to be broadcast to input shape (batch dims)
    # Select auto-broadcasts usually.
    return ops.select(mask, inp, zero)

@OpRegistry.register_function(operator.getitem)
def convert_getitem(builder, node, args, kwargs):
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
                 # Tensor indexing (not full gather yet, assuming slice logic)
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
        # Tensor indexing -> Gather (along axis 0 by default for single index)
        idx_node = builder.get_input(idx.name)
        axis = ops.constant([0], dtype=np.int64)
        return ops.gather(inp, idx_node, axis)

    raise NotImplementedError(f"getitem index {type(idx)} not implemented")

# --- Module Converters ---

@OpRegistry.register_module(torch.nn.Linear)
def convert_linear_module(builder, node, submod, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    
    mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)
    res = mm
    if submod.bias is not None:
        b_const = builder.add_constant(f"{node.target}.bias", submod.bias)
        res = ops.add(mm, b_const)
    return res

@OpRegistry.register_module(torch.nn.Embedding)
def convert_embedding_module(builder, node, submod, args, kwargs):
    indices = builder.get_input_or_constant(args[0])
    w_const = builder.add_constant(f"{node.target}.weight", submod.weight)
    axis = ops.constant([0], dtype=np.int64)
    return ops.gather(w_const, indices, axis)

@OpRegistry.register_module(torch.nn.ReLU)
def convert_relu_module(builder, node, submod, args, kwargs):
    return ops.relu(builder.get_input_or_constant(args[0]))

@OpRegistry.register_module(torch.nn.GELU)
def convert_gelu_module(builder, node, submod, args, kwargs):
    return ops.gelu(builder.get_input_or_constant(args[0]), approximation_mode="erf")

@OpRegistry.register_module(torch.nn.SiLU)
def convert_silu_module(builder, node, submod, args, kwargs):
    return ops.swish(builder.get_input_or_constant(args[0]))

@OpRegistry.register_module(torch.nn.Softmax)
def convert_softmax_module(builder, node, submod, args, kwargs):
    dim = submod.dim if submod.dim is not None else -1
    return ops.softmax(builder.get_input_or_constant(args[0]), axis=dim)

# --- Method Converters ---

@OpRegistry.register_method("contiguous", "type_as", "to")
def convert_noop(builder, node, args, kwargs):
    return builder.get_input_or_constant(args[0])

@OpRegistry.register_method("float")
def convert_to_float(builder, node, args, kwargs):
    inp = builder.get_input_or_constant(args[0])
    return ops.convert(inp, destination_type='f32')

@OpRegistry.register_method("flatten")
def convert_flatten(builder, node, args, kwargs):
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


# --- Main Compiler Logic ---

class NPUGraphModule(torch.nn.Module):
    def __init__(self, compiled_model, input_names):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.infer_request = self.compiled_model.create_infer_request()

    def forward(self, *args):
        # Map inputs
        # We assume args are in the same order as parameters were created.
        # Zero-copy passing
        for i, val in enumerate(args):
            # Determine expected type from compiled model input
            ov_input = self.compiled_model.inputs[i]
            ov_type = ov_input.get_element_type()
            
            # Map OpenVINO type to numpy type
            target_dtype = np.float32
            
            if ov_type == ov.Type.f32:
                target_dtype = np.float32
            elif ov_type == ov.Type.f16:
                target_dtype = np.float16
            elif ov_type == ov.Type.i32:
                target_dtype = np.int32
            elif ov_type == ov.Type.i64:
                target_dtype = np.int64
            elif ov_type == ov.Type.i8:
                target_dtype = np.int8
            elif ov_type == ov.Type.u8:
                target_dtype = np.uint8
            elif ov_type == ov.Type.boolean:
                target_dtype = bool
            else:
                # Fallback to string check if direct comparison fails (unlikely)
                str_type = str(ov_type)
                if 'i32' in str_type: target_dtype = np.int32
                elif 'i64' in str_type: target_dtype = np.int64
                elif 'i8' in str_type: target_dtype = np.int8
                elif 'f16' in str_type: target_dtype = np.float16
            
            # Ensure contiguous CPU numpy array
            if isinstance(val, torch.Tensor):
                np_view = val.detach().cpu().numpy()
            elif isinstance(val, (int, float)):
                np_view = np.array(val)
            else:
                np_view = np.array(val)
            
            # Check and convert if mismatch
            if np_view.dtype != target_dtype:
                np_view = np_view.astype(target_dtype)
            
            # set_input_tensor can take index or name
            self.infer_request.set_input_tensor(i, ov.Tensor(np_view, shared_memory=True))
        
        self.infer_request.infer()
        
        # Retrieve outputs
        outputs = []
        for j in range(len(self.compiled_model.outputs)):
            out_tensor = self.infer_request.get_output_tensor(j)
            outputs.append(torch.from_numpy(out_tensor.data).clone())
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

def compile_to_npu(model: torch.nn.Module, example_input: Any) -> torch.nn.Module:
    # 1. Trace
    traced = torch.fx.symbolic_trace(model)
    
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)

    # 2. Capture Values (Interpreter)
    interpreter = ValueCapturingInterpreter(traced)
    # Run interpreter to capture values of all nodes
    # Note: This executes the model on CPU once!
    interpreter.run(*example_input)
    
    # 3. Build OV Graph
    builder = OVGraphBuilder(interpreter.node_values)
    
    # Propagate shapes (still useful for tensor meta)
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(*example_input) 
    
    input_iter = iter(example_input)
    
    for node in traced.graph.nodes:
        # print(f"Processing node: {node.name}, op: {node.op}, target: {node.target}")
        
        if node.op == 'placeholder':
            try:
                val = next(input_iter)
            except StopIteration:
                 raise RuntimeError(f"Not enough example inputs for placeholders starting at {node.name}")

            if 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                dtype = node.meta['tensor_meta'].dtype
            else:
                if isinstance(val, torch.Tensor):
                    shape = list(val.shape); dtype = val.dtype
                elif isinstance(val, int):
                    shape = []; dtype = torch.int64
                elif isinstance(val, float):
                    shape = []; dtype = torch.float32
                else:
                    print(f"Warning: Unknown type for placeholder {node.name}, defaulting to float32 [1]")
                    shape = [1]; dtype = torch.float32

            builder.add_parameter(node.name, list(shape), dtype)
            
        elif node.op == 'call_function':
            converter = OpRegistry.get_function(node.target)
            if converter:
                res = converter(builder, node, node.args, node.kwargs)
                builder.register_output(node.name, res)
            else:
                raise NotImplementedError(f"Function {node.target} not implemented")

        elif node.op == 'call_method':
            converter = OpRegistry.get_method(node.target)
            if converter:
                res = converter(builder, node, node.args, node.kwargs)
                builder.register_output(node.name, res)
            else:
                raise NotImplementedError(f"Method {node.target} not implemented")

        elif node.op == 'call_module':
            submod = model
            for atom in node.target.split('.'):
                submod = getattr(submod, atom)
            
            converter = OpRegistry.get_module(type(submod))
            if converter:
                res = converter(builder, node, submod, node.args, node.kwargs)
                builder.register_output(node.name, res)
            else:
                raise NotImplementedError(f"Module {type(submod)} not implemented")

        elif node.op == 'get_attr':
            atom = model
            for atom_name in node.target.split('.'):
                atom = getattr(atom, atom_name)
            builder.add_constant(node.name, atom)
            
        elif node.op == 'output':
            ret_vals = node.args[0]
            if isinstance(ret_vals, tuple):
                for ret_val in ret_vals:
                    builder.result_nodes.append(builder.get_input(ret_val.name))
            else:
                builder.result_nodes.append(builder.get_input(ret_vals.name))

    # 3. Create OV Model
    ov_model = ov.Model(builder.result_nodes, builder.parameters, "NPU_Model")
    
    # 4. Compile
    core = ov.Core()
    compiled = core.compile_model(ov_model, "NPU")
    
    # 5. Wrap
    return NPUGraphModule(compiled, [p.friendly_name for p in builder.parameters])


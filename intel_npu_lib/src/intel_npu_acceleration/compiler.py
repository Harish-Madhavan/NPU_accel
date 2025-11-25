
import torch
import torch.nn as nn
import torch.fx
import openvino as ov
import openvino.runtime.opset13 as ops
import numpy as np
from typing import Dict, Any, List
import operator
import builtins
from . import transpose as npu_transpose_func
from . import reshape as npu_reshape_func
from . import rmsnorm as npu_rmsnorm_func

class OVGraphBuilder:
    def __init__(self):
        self.node_map: Dict[str, Any] = {} # Maps fx node name to OV node output
        self.parameters: List[Any] = []
        self.result_nodes: List[Any] = [] # List of OV nodes that are outputs
        
    def get_input(self, node_name: str):
        if node_name not in self.node_map:
            raise RuntimeError(f"Node {node_name} not found in graph map.")
        return self.node_map[node_name]

    def add_parameter(self, name: str, shape: List[int], dtype):
        # Convert torch dtype to numpy dtype
        if dtype == torch.float32: np_type = np.float32
        elif dtype == torch.int64: np_type = np.int64
        else: np_type = np.float32 # Default
        
        # OpenVINO expects static shapes for compilation usually, or dynamic with bounds.
        # For now, let's assume static shapes based on example input.
        param = ops.parameter(ov.Shape(shape), dtype=np_type, name=name)
        self.node_map[name] = param
        self.parameters.append(param)
        return param

    def add_constant(self, name: str, tensor: torch.Tensor):
        # Tensor to numpy
        data = tensor.detach().cpu().numpy()
        const_node = ops.constant(data, name=name)
        self.node_map[name] = const_node
        return const_node

    def register_output(self, name: str, ov_node):
        # print(f"DEBUG: Registering output for {name}")
        self.node_map[name] = ov_node

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
            # Ensure contiguous CPU numpy array
            # If tensor is on GPU, this will copy. 
            # If on CPU and contiguous, it shares.
            if isinstance(val, torch.Tensor):
                np_view = val.detach().cpu().numpy()
            elif isinstance(val, (int, float)):
                # OpenVINO expects array for scalar inputs too if parameter is rank 0 or 1
                # Our parameters are usually static shape.
                # If we defined parameter as scalar (rank 0), np.array(val) works.
                # If rank 1 [1], np.array([val]) works.
                # In compilation, we used shape [1] for scalars usually.
                # Let's check how we defined parameters in 'compile_to_npu'
                # We used shape=[1] for int/float placeholders.
                np_view = np.array([val], dtype=np.int64 if isinstance(val, int) else np.float32)
            else:
                np_view = np.array(val)
            
            # set_input_tensor can take index or name
            # Note: shared_memory=True requires contiguous memory and matching types.
            # Creating new numpy array from int/float is safe.
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
    
    # 2. Build OV Graph
    builder = OVGraphBuilder()
    
    # Propagate shapes (we need them for Parameters)
    # Ensure example_input is a tuple
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)
        
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(*example_input) 
    
    # Map placeholders to inputs
    input_iter = iter(example_input)
    
    for node in traced.graph.nodes:
        # Debug
        # print(f"Processing node: {node.name}, op: {node.op}, target: {node.target}")
        if node.op == 'placeholder':
            # Always advance input iterator to stay in sync
            try:
                val = next(input_iter)
            except StopIteration:
                 raise RuntimeError(f"Not enough example inputs for placeholders starting at {node.name}")

            # Input
            if 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                dtype = node.meta['tensor_meta'].dtype
            else:
                # Fallback to example input
                if isinstance(val, torch.Tensor):
                    shape = list(val.shape)
                    dtype = val.dtype
                elif isinstance(val, int):
                    shape = [1]
                    dtype = torch.int64
                elif isinstance(val, float):
                    shape = [1]
                    dtype = torch.float32
                else:
                    # Default/Unknown
                    print(f"Warning: Unknown type for placeholder {node.name}, defaulting to float32 [1]")
                    shape = [1]
                    dtype = torch.float32

            builder.add_parameter(node.name, list(shape), dtype)
            
        elif node.op == 'call_function':
            target = node.target
            args = node.args
            
            if target == torch.add or target == operator.add:
                # Inputs can be nodes or constants (scalars)
                arg0 = args[0]
                arg1 = args[1]
                
                if isinstance(arg0, (int, float)):
                    inp0 = ops.constant(arg0, dtype=np.float32)
                else:
                    inp0 = builder.get_input(arg0.name)
                    
                if isinstance(arg1, (int, float)):
                    inp1 = ops.constant(arg1, dtype=np.float32)
                else:
                    inp1 = builder.get_input(arg1.name)
                    
                res = ops.add(inp0, inp1)
                builder.register_output(node.name, res)
                
            elif target == builtins.getattr:
                # args: (obj, name)
                obj_node = args[0]
                attr_name = args[1]
                if attr_name == 'shape':
                    # Extract static shape from meta
                    if 'tensor_meta' in obj_node.meta:
                        shape = list(obj_node.meta['tensor_meta'].shape)
                        res = ops.constant(np.array(shape, dtype=np.int64))
                        builder.register_output(node.name, res)
                    else:
                        raise RuntimeError(f"Cannot get shape for {obj_node.name}: no meta info")
                else:
                    raise NotImplementedError(f"getattr({attr_name}) not implemented")

            elif target == operator.getitem:
                inp = builder.get_input(args[0].name)
                idx = args[1]
                
                if isinstance(idx, int):
                    indices = ops.constant([idx], dtype=np.int64)
                    axis = ops.constant([0], dtype=np.int64)
                    gather = ops.gather(inp, indices, axis)
                    res = ops.squeeze(gather, axis)
                    builder.register_output(node.name, res)
                
                elif isinstance(idx, (slice, tuple)):
                    if isinstance(idx, slice):
                        idx = (idx,)
                    
                    begin_nodes = []
                    end_nodes = []
                    strides_nodes = []
                    begin_mask = []
                    end_mask = []
                    ellipsis_mask = []
                    new_axis_mask = []
                    shrink_axis_mask = []
                    
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
                            elif isinstance(item.start, int):
                                start_val = ops.constant([item.start], dtype=np.int64)
                                begin_mask.append(0)
                            else:
                                v = builder.get_input(item.start.name)
                                start_val = ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False)
                                begin_mask.append(0)
                            
                            if item.stop is None:
                                stop_val = ops.constant([0], dtype=np.int64)
                                end_mask.append(1)
                            elif isinstance(item.stop, int):
                                stop_val = ops.constant([item.stop], dtype=np.int64)
                                end_mask.append(0)
                            else:
                                v = builder.get_input(item.stop.name)
                                stop_val = ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False)
                                end_mask.append(0)
                                
                            if item.step is None:
                                step_val = ops.constant([1], dtype=np.int64)
                            elif isinstance(item.step, int):
                                step_val = ops.constant([item.step], dtype=np.int64)
                            else:
                                v = builder.get_input(item.step.name)
                                step_val = ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False)
                            
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
                             start_val = ops.reshape(v, ops.constant([1], dtype=np.int64), special_zero=False)
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
                    
                    res = ops.strided_slice(inp, begin_t, end_t, strides_t, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask)
                    builder.register_output(node.name, res)

                else:
                     raise NotImplementedError(f"getitem index {type(idx)} not implemented")

            elif target == torch.mul or target == operator.mul:
                arg0 = args[0]
                arg1 = args[1]
                
                inp0 = ops.constant(arg0, dtype=np.int64 if isinstance(arg0, int) else np.float32) if isinstance(arg0, (int, float)) else builder.get_input(arg0.name)
                inp1 = ops.constant(arg1, dtype=np.int64 if isinstance(arg1, int) else np.float32) if isinstance(arg1, (int, float)) else builder.get_input(arg1.name)
                
                if inp0.get_element_type() != inp1.get_element_type():
                    if 'f' in str(inp0.get_element_type()) or 'f' in str(inp1.get_element_type()):
                         inp0 = ops.convert(inp0, destination_type=np.float32)
                         inp1 = ops.convert(inp1, destination_type=np.float32)
                    else:
                         inp0 = ops.convert(inp0, destination_type=np.int64)
                         inp1 = ops.convert(inp1, destination_type=np.int64)

                res = ops.multiply(inp0, inp1)
                builder.register_output(node.name, res)

            elif target == torch.sub or target == operator.sub:
                arg0 = args[0]
                arg1 = args[1]
                
                inp0 = ops.constant(arg0, dtype=np.int64 if isinstance(arg0, int) else np.float32) if isinstance(arg0, (int, float)) else builder.get_input(arg0.name)
                inp1 = ops.constant(arg1, dtype=np.int64 if isinstance(arg1, int) else np.float32) if isinstance(arg1, (int, float)) else builder.get_input(arg1.name)
                
                if inp0.get_element_type() != inp1.get_element_type():
                    if 'f' in str(inp0.get_element_type()) or 'f' in str(inp1.get_element_type()):
                         inp0 = ops.convert(inp0, destination_type=np.float32)
                         inp1 = ops.convert(inp1, destination_type=np.float32)
                    else:
                         inp0 = ops.convert(inp0, destination_type=np.int64)
                         inp1 = ops.convert(inp1, destination_type=np.int64)
                    
                res = ops.subtract(inp0, inp1)
                builder.register_output(node.name, res)

            elif target == torch.div or target == operator.truediv:
                arg0 = args[0]
                arg1 = args[1]
                
                inp0 = ops.constant(arg0, dtype=np.int64 if isinstance(arg0, int) else np.float32) if isinstance(arg0, (int, float)) else builder.get_input(arg0.name)
                inp1 = ops.constant(arg1, dtype=np.int64 if isinstance(arg1, int) else np.float32) if isinstance(arg1, (int, float)) else builder.get_input(arg1.name)
                
                if inp0.get_element_type() != inp1.get_element_type():
                    if 'f' in str(inp0.get_element_type()) or 'f' in str(inp1.get_element_type()):
                         inp0 = ops.convert(inp0, destination_type=np.float32)
                         inp1 = ops.convert(inp1, destination_type=np.float32)
                    else:
                         inp0 = ops.convert(inp0, destination_type=np.float32) # Division usually promotes to float
                         inp1 = ops.convert(inp1, destination_type=np.float32)

                res = ops.divide(inp0, inp1)
                builder.register_output(node.name, res)

            elif target == operator.floordiv:
                arg0 = args[0]
                arg1 = args[1]
                
                inp0 = ops.constant(arg0, dtype=np.int64 if isinstance(arg0, int) else np.float32) if isinstance(arg0, (int, float)) else builder.get_input(arg0.name)
                inp1 = ops.constant(arg1, dtype=np.int64 if isinstance(arg1, int) else np.float32) if isinstance(arg1, (int, float)) else builder.get_input(arg1.name)
                
                if inp0.get_element_type() != inp1.get_element_type():
                    if 'f' in str(inp0.get_element_type()) or 'f' in str(inp1.get_element_type()):
                         inp0 = ops.convert(inp0, destination_type=np.float32)
                         inp1 = ops.convert(inp1, destination_type=np.float32)
                    else:
                         inp0 = ops.convert(inp0, destination_type=np.int64)
                         inp1 = ops.convert(inp1, destination_type=np.int64)
                
                div = ops.divide(inp0, inp1)
                res = ops.floor(div)
                res = ops.convert(res, destination_type=np.int64)
                builder.register_output(node.name, res)

            elif target == torch.matmul or target == torch.mm:
                inp0 = builder.get_input(args[0].name)
                inp1 = builder.get_input(args[1].name)
                res = ops.matmul(inp0, inp1, transpose_a=False, transpose_b=False)
                builder.register_output(node.name, res)
                
            elif target == torch.relu or target == torch.nn.functional.relu:
                inp = builder.get_input(args[0].name)
                res = ops.relu(inp)
                builder.register_output(node.name, res)
                
            elif target == torch.nn.functional.gelu:
                inp = builder.get_input(args[0].name)
                # Gelu "erf" mode
                res = ops.gelu(inp, approximation_mode="erf")
                builder.register_output(node.name, res)
                
            elif target == torch.nn.functional.silu:
                inp = builder.get_input(args[0].name)
                res = ops.swish(inp)
                builder.register_output(node.name, res)
            
            elif target == torch.pow or target == operator.pow:
                inp = builder.get_input(args[0].name)
                exponent = args[1]
                # Exponent can be a constant number or a node
                if isinstance(exponent, (int, float)):
                    exp_node = ops.constant(exponent, dtype=np.float32)
                else:
                    exp_node = builder.get_input(exponent.name)
                res = ops.power(inp, exp_node)
                builder.register_output(node.name, res)

            elif target == torch.mean:
                inp = builder.get_input(args[0].name)
                dim = node.kwargs.get('dim', args[1] if len(args) > 1 else None)
                keepdim = node.kwargs.get('keepdim', args[2] if len(args) > 2 else False)
                
                # Dim can be int or tuple
                if dim is None:
                    # Reduce all?
                    # OV needs axes.
                    # For now assume we always have dim
                    raise RuntimeError("Mean without dim not supported yet")
                
                if isinstance(dim, int):
                    axes = ops.constant([dim], dtype=np.int64)
                else:
                    axes = ops.constant(list(dim), dtype=np.int64)
                    
                res = ops.reduce_mean(inp, axes, keep_dims=keepdim)
                builder.register_output(node.name, res)

            elif target == torch.rsqrt:
                inp = builder.get_input(args[0].name)
                # rsqrt(x) = x^(-0.5) or 1/sqrt(x)
                # OV has no direct rsqrt in standard opset usually? Check.
                # opset1/4 don't have it? 
                # We can use power(x, -0.5)
                exp_node = ops.constant(-0.5, dtype=np.float32)
                res = ops.power(inp, exp_node)
                builder.register_output(node.name, res)

            elif target == torch.cat:
                # args: ([tensors],), kwargs: {'dim': dim}
                tensors_list_node = node.args[0] # This is a list of node names
                dim = node.kwargs['dim']
                
                # Iterate through the tuple of nodes to get OV nodes
                ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
                
                res = ops.concat(ov_inputs, axis=dim)
                builder.register_output(node.name, res)

            elif target == torch.stack:
                # args: ([tensors],), kwargs: {'dim': dim}
                tensors_list_node = node.args[0]
                dim = node.kwargs['dim']
                
                ov_inputs = [builder.get_input(n.name) for n in tensors_list_node]
                
                # Implement stack using unsqueeze and concat
                unsqueezed_inputs = []
                for ov_input in ov_inputs:
                    # Unsqueeze adds a new dimension at 'dim'
                    unsqueezed_inputs.append(ops.unsqueeze(ov_input, ops.constant(np.array([dim]), dtype=np.int64)))
                
                res = ops.concat(unsqueezed_inputs, axis=dim)
                builder.register_output(node.name, res)

            elif target == torch.softmax or target == torch.nn.functional.softmax:
                inp = builder.get_input(args[0].name)
                dim = node.kwargs.get('dim', args[1] if len(args) > 1 else -1)
                res = ops.softmax(inp, axis=dim)
                builder.register_output(node.name, res)

            elif target == torch.transpose or target == npu_transpose_func:
                inp = builder.get_input(args[0].name)
                dim0 = args[1]
                dim1 = args[2]
                # We need rank to construct permutation. 
                # We can get input rank from tensor_meta if available, or from OV node partial shape?
                # OV node partial shape is available.
                rank = inp.get_output_partial_shape(0).rank.get_length()
                perm = list(range(rank))
                if dim0 < 0: dim0 += rank
                if dim1 < 0: dim1 += rank
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                
                res = ops.transpose(inp, perm)
                builder.register_output(node.name, res)

            elif target == torch.reshape or target == npu_reshape_func:
                inp = builder.get_input(args[0].name)
                # Shape can be args[1] (tuple) or args[1:] (varargs)
                # In FX, it depends on how it was called.
                req_shape = args[1]
                if isinstance(req_shape, (tuple, list)):
                    # It's a list/tuple
                    pass
                else:
                    # It's separate args
                    req_shape = args[1:]
                
                # Normalize shape to list of ints
                # Note: if shape contains -1, OV Reshape supports it (special_zero=False handles 0 as 0, -1 as infer)
                # But we need to ensure it's passed as a Constant node
                res = ops.reshape(inp, list(req_shape), special_zero=False)
                builder.register_output(node.name, res)

            elif target == npu_rmsnorm_func:
                # Args: input, weight, eps
                inp = builder.get_input(args[0].name)
                
                # Weight might be a constant from a parameter
                # If args[1] is a Node, get it.
                w_node = args[1]
                weight = builder.get_input(w_node.name)
                
                eps = args[2]
                
                # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
                x_sq = ops.multiply(inp, inp)
                
                # Reduce mean over last dim
                # Get rank
                rank = inp.get_output_partial_shape(0).rank.get_length()
                axes = ops.constant([rank - 1], dtype=np.int64)
                
                mean_sq = ops.reduce_mean(x_sq, axes, keep_dims=True)
                
                eps_const = ops.constant(eps, dtype=np.float32)
                variance = ops.add(mean_sq, eps_const)
                std_dev = ops.sqrt(variance)
                x_norm = ops.divide(inp, std_dev)
                res = ops.multiply(x_norm, weight)
                
                builder.register_output(node.name, res)

            elif target == torch.nn.functional.linear:
                # args: input, weight, bias(optional)
                # Note: FX graph for functional.linear usually has inputs as nodes.
                # If weights are constants (from params), they might be 'get_attr' nodes?
                # Wait, symbolic_trace keeps parameters as 'get_attr' usually, 
                # BUT functional.linear takes tensors. 
                # If we trace a module, the weights are attributes of the module.
                # FX usually unpacks this.
                pass # Handled in general logic below? No, specific handling needed.
                
                # Input
                inp = builder.get_input(args[0].name)
                
                # Weight
                # In FX, if weight is a param, it comes from a previous node?
                # Or is it an arg that is a Node object?
                w_node = args[1]
                w_val = builder.get_input(w_node.name)
                
                # Linear in OV: MatMul(x, w^T) + bias
                # PyTorch Linear weight is (Out, In). MatMul expects (In, Out) for second arg if not transposed.
                # We can use MatMul with transpose_b=True
                mm = ops.matmul(inp, w_val, transpose_a=False, transpose_b=True)
                
                res = mm
                if submod.bias is not None:
                    b_const = builder.add_constant(b_name, submod.bias)
                    res = ops.add(mm, b_const)
                
                builder.register_output(node.name, res)
                print(f"DEBUG: Registered output for Linear module: {node.name}")

            else:
                print(f"DEBUG: Unhandled function: {target} for {node.name}")
                raise NotImplementedError(f"Function {target} not implemented in NPU Graph Compiler")

        elif node.op == 'call_module':
            submod_name = node.target
            submod = model
            for atom in submod_name.split('.'):
                submod = getattr(submod, atom)
            
            args = node.args
            
            if isinstance(submod, torch.nn.Linear):
                inp = builder.get_input(args[0].name)
                
                w_name = f"{submod_name}.weight"
                w_const = builder.add_constant(w_name, submod.weight)
                
                mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)
                
                res = mm
                if submod.bias is not None:
                    b_name = f"{submod_name}.bias"
                    b_const = builder.add_constant(b_name, submod.bias)
                    res = ops.add(mm, b_const)
                
                builder.register_output(node.name, res)
            
            elif isinstance(submod, torch.nn.Embedding):
                 indices = builder.get_input(args[0].name)
                 w_name = f"{submod_name}.weight"
                 w_const = builder.add_constant(w_name, submod.weight)
                 
                 axis = ops.constant([0], dtype=np.int64)
                 res = ops.gather(w_const, indices, axis)
                 builder.register_output(node.name, res)

            else:
                raise NotImplementedError(f"Module {type(submod)} not implemented in NPU Graph Compiler")

        elif node.op == 'call_method':
            method = node.target
            args = node.args
            
            if method == 'mean':
                inp = builder.get_input(args[0].name)
                dim = node.kwargs.get('dim', args[1] if len(args) > 1 else None)
                keepdim = node.kwargs.get('keepdim', args[2] if len(args) > 2 else False)
                
                if isinstance(dim, int):
                    axes = ops.constant([dim], dtype=np.int64)
                elif isinstance(dim, (tuple, list)):
                    axes = ops.constant(list(dim), dtype=np.int64)
                else:
                    # Default reduction
                    # Get rank? Or reduce all.
                    # For RMSNorm it's usually -1.
                    # If dim is missing, we might crash.
                    raise RuntimeError(f"Mean method requires dim")

                res = ops.reduce_mean(inp, axes, keep_dims=keepdim)
                builder.register_output(node.name, res)
                
            elif method == 'pow':
                inp = builder.get_input(args[0].name)
                exponent = args[1]
                if isinstance(exponent, (int, float)):
                    exp_node = ops.constant(exponent, dtype=np.float32)
                else:
                    exp_node = builder.get_input(exponent.name)
                res = ops.power(inp, exp_node)
                builder.register_output(node.name, res)
                
            elif method == 'view' or method == 'reshape':
                inp = builder.get_input(args[0].name)
                req_shape = args[1]
                
                if isinstance(req_shape, (tuple, list)):
                    shape_items = req_shape
                else:
                    shape_items = args[1:]
                
                shape_nodes = []
                for item in shape_items:
                    if isinstance(item, int):
                        shape_nodes.append(ops.constant([item], dtype=np.int64))
                    elif hasattr(item, 'name'):
                        val_node = builder.get_input(item.name)
                        # Reshape to [1] to ensure 1D for concat
                        shape_nodes.append(ops.reshape(val_node, ops.constant([1], dtype=np.int64), special_zero=False))
                    else:
                        raise RuntimeError(f"Unknown shape item type: {type(item)}")
                
                shape_tensor = ops.concat(shape_nodes, axis=0)
                res = ops.reshape(inp, shape_tensor, special_zero=False)
                builder.register_output(node.name, res)
                
            elif method == 'transpose':
                inp = builder.get_input(args[0].name)
                dim0 = args[1]
                dim1 = args[2]
                rank = inp.get_output_partial_shape(0).rank.get_length()
                perm = list(range(rank))
                if dim0 < 0: dim0 += rank
                if dim1 < 0: dim1 += rank
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                res = ops.transpose(inp, perm)
                builder.register_output(node.name, res)
                
            elif method == 'contiguous':
                # No-op
                inp = builder.get_input(args[0].name)
                builder.register_output(node.name, inp)
                
            elif method == 'flatten':
                inp = builder.get_input(args[0].name)
                # flatten(start_dim, end_dim)
                # Default flatten(0, -1) -> 1D
                # OV Reshape to [-1] if fully flattened?
                # Or use generic reshape.
                # Simplest: reshape to [d0, ..., d_start-1, -1, d_end+1, ...]
                # For now assume flatten(start_dim=?)
                # If generic flatten, use reshape(..., -1) if simple
                # Let's implement simple flatten(2, 3) -> merge dims
                # Or Flatten op? OV has Flatten? No, Reshape.
                # Let's assume flatten(3) -> from 3 to end.
                start_dim = args[1] if len(args) > 1 else 0
                end_dim = args[2] if len(args) > 2 else -1
                
                # For simple cases (like flattening last 2 dims):
                # Just allow Reshape to handle it if possible, but we need target shape.
                # Without shape inference helper, hard to know target shape.
                # But wait, if it's used in `apply_rotary_emb` -> `.flatten(3)`
                # (B, S, H, D/2, 2) -> flatten(3) -> (B, S, H, D)
                # This merges last two dimensions.
                # We can use Reshape.
                # But we need to know dimensions 0, 1, 2.
                # We can use 0 (copy) for preserved dims.
                # reshape(inp, [0, 0, 0, -1], special_zero=True)
                # 0 copies dimension from input.
                
                # If flatten(3), means merge 3 and 4.
                # We can construct shape pattern: [0, 0, 0, -1]
                # Assuming rank is 5.
                
                # Getting rank
                rank = inp.get_output_partial_shape(0).rank.get_length()
                if start_dim < 0: start_dim += rank
                if end_dim < 0: end_dim += rank
                
                shape_pattern = []
                for i in range(start_dim):
                    shape_pattern.append(0) # Copy
                shape_pattern.append(-1) # Merge
                for i in range(end_dim + 1, rank):
                    shape_pattern.append(0) # Copy
                    
                res = ops.reshape(inp, shape_pattern, special_zero=True)
                builder.register_output(node.name, res)
                
            elif method == 'type_as':
                # Cast?
                # If x.type_as(y), we need dtype of y.
                # But y is a node.
                # For compilation, we might assume float32 everywhere for now.
                # Just no-op
                inp = builder.get_input(args[0].name)
                builder.register_output(node.name, inp)
                
            elif method == 'float':
                inp = builder.get_input(args[0].name)
                res = ops.convert(inp, destination_type='f32')
                builder.register_output(node.name, res)
                
            elif method == 'to':
                # x.to(...)
                # Could be device or dtype
                # If args[1] is dtype, cast.
                # If args[1] is tensor, like x.to(y), cast to y.dtype?
                # For now, assume simple cast or no-op if device move.
                # Let's treat as no-op for device moves, convert for dtype if possible.
                # But arg might be torch.float32.
                inp = builder.get_input(args[0].name)
                # Simplification: No-op
                builder.register_output(node.name, inp)
                
            else:
                raise NotImplementedError(f"Method {method} not implemented")

        elif node.op == 'get_attr':
            # Parameter/Buffer access
            # Fetch data from model
            atom = model
            for atom_name in node.target.split('.'):
                atom = getattr(atom, atom_name)
            
            # 'atom' is now the tensor (weight/bias)
            builder.add_constant(node.name, atom)
            
        elif node.op == 'output':
            # Result
            # args[0] is the return value (could be tuple)
            args = node.args
            ret_vals = args[0]

            if isinstance(ret_vals, tuple):
                for ret_val in ret_vals:
                    builder.result_nodes.append(builder.get_input(ret_val.name))
            else:
                builder.result_nodes.append(builder.get_input(ret_vals.name))

    # 3. Create OV Model
    # OpenVINO model constructor: ov.Model(OutputVector, ParameterVector, model_name)
    # OutputVector can be a single Output or a list of Outputs
    ov_model = ov.Model(builder.result_nodes, builder.parameters, "NPU_Model")
    
    # 4. Compile
    core = ov.Core()
    compiled = core.compile_model(ov_model, "NPU")
    
    # 5. Wrap
    return NPUGraphModule(compiled, [p.friendly_name for p in builder.parameters])


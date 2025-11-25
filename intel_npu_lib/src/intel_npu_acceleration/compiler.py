
import torch
import torch.nn as nn
import torch.fx
import openvino as ov
import openvino.runtime.opset13 as ops
import numpy as np
from typing import Dict, Any, List
import operator
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
            np_view = val.detach().cpu().numpy()
            
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
    
    # 2. Build OV Graph
    builder = OVGraphBuilder()
    
    # Propagate shapes (we need them for Parameters)
    # Ensure example_input is a tuple
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)
        
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(*example_input) 
    
    for node in traced.graph.nodes:
        if node.op == 'placeholder':
            # Input
            shape = node.meta['tensor_meta'].shape
            dtype = node.meta['tensor_meta'].dtype
            builder.add_parameter(node.name, list(shape), dtype)
            
        elif node.op == 'call_function':
            target = node.target
            args = node.args
            
            if target == torch.add or target == operator.add:
                inp0 = builder.get_input(args[0].name)
                inp1 = builder.get_input(args[1].name)
                res = ops.add(inp0, inp1)
                builder.register_output(node.name, res)
                
            elif target == torch.matmul or target == torch.mm:
                inp0 = builder.get_input(args[0].name)
                inp1 = builder.get_input(args[1].name)
                res = ops.matmul(inp0, inp1)
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
                if len(args) > 2 and args[2] is not None:
                    b_node = args[2]
                    b_val = builder.get_input(b_node.name)
                    res = ops.add(mm, b_val)
                
                builder.register_output(node.name, res)

            else:
                raise NotImplementedError(f"Op {target} not implemented in NPU Graph Compiler")

        elif node.op == 'call_module':
            submod = model.get_submodule(node.target)
            args = node.args
            
            if isinstance(submod, torch.nn.Linear):
                # Input
                inp = builder.get_input(args[0].name)
                
                # Weights are attributes of the module
                # We create constants for them.
                # Name them uniquely to avoid collision? 
                # Use node.name + .weight
                w_name = f"{node.name}.weight"
                b_name = f"{node.name}.bias"
                
                w_const = builder.add_constant(w_name, submod.weight)
                
                # Linear: MatMul(x, w^T)
                mm = ops.matmul(inp, w_const, transpose_a=False, transpose_b=True)
                res = mm
                
                if submod.bias is not None:
                    b_const = builder.add_constant(b_name, submod.bias)
                    res = ops.add(mm, b_const)
                
                builder.register_output(node.name, res)
                
            elif isinstance(submod, torch.nn.ReLU):
                inp = builder.get_input(args[0].name)
                res = ops.relu(inp)
                builder.register_output(node.name, res)
                
            elif isinstance(submod, torch.nn.GELU):
                inp = builder.get_input(args[0].name)
                res = ops.gelu(inp, approximation_mode="erf")
                builder.register_output(node.name, res)
                
            elif isinstance(submod, torch.nn.SiLU):
                inp = builder.get_input(args[0].name)
                res = ops.swish(inp)
                builder.register_output(node.name, res)
            else:
                raise NotImplementedError(f"Module {type(submod)} not implemented in NPU Graph Compiler")

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


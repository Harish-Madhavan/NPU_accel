import torch
import torch.fx
import openvino as ov
import numpy as np
from typing import Any, List, Optional, Tuple, Union
import logging
from .registry import OpRegistry
from .graph_builder import OVGraphBuilder, ValueCapturingInterpreter
# Import converters to ensure they are registered
from . import converters 

logger = logging.getLogger(__name__)

class NPUCompilationError(Exception):
    """Exception raised for errors during NPU compilation."""
    pass

class NPUGraphModule(torch.nn.Module):
    def __init__(self, compiled_model, input_names):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_names = input_names
        self.infer_request = self.compiled_model.create_infer_request()

    def forward(self, *args):
        # Map inputs
        for i, val in enumerate(args):
            ov_input = self.compiled_model.inputs[i]
            ov_type = ov_input.get_element_type()
            
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
                str_type = str(ov_type)
                if 'i32' in str_type: target_dtype = np.int32
                elif 'i64' in str_type: target_dtype = np.int64
                elif 'i8' in str_type: target_dtype = np.int8
                elif 'f16' in str_type: target_dtype = np.float16
            
            if isinstance(val, torch.Tensor):
                np_view = val.detach().cpu().numpy()
            elif isinstance(val, (int, float)):
                np_view = np.array(val)
            else:
                np_view = np.array(val)
            
            if np_view.dtype != target_dtype:
                np_view = np_view.astype(target_dtype)
            
            self.infer_request.set_input_tensor(i, ov.Tensor(np_view, shared_memory=True))
        
        self.infer_request.infer()
        
        outputs = []
        for j in range(len(self.compiled_model.outputs)):
            out_tensor = self.infer_request.get_output_tensor(j)
            outputs.append(torch.from_numpy(out_tensor.data).clone())
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

def compile_to_npu(model: torch.nn.Module, example_input: Any) -> torch.nn.Module:
    logger.info("Starting NPU Compilation...")
    try:
        # 1. Trace
        traced = torch.fx.symbolic_trace(model)
        
        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)

        # 2. Capture Values (Interpreter)
        interpreter = ValueCapturingInterpreter(traced)
        interpreter.run(*example_input)
        
        # 3. Build OV Graph
        builder = OVGraphBuilder(interpreter.node_values)
        
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(traced).propagate(*example_input) 
        
        input_iter = iter(example_input)
        
        for node in traced.graph.nodes:
            # logger.debug(f"Processing node: {node.name}, op: {node.op}, target: {node.target}")
            
            if node.op == 'placeholder':
                try:
                    val = next(input_iter)
                except StopIteration:
                    raise NPUCompilationError(f"Not enough example inputs for placeholders starting at {node.name}")

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
                        logger.warning(f"Unknown type for placeholder {node.name}, defaulting to float32 [1]")
                        shape = [1]; dtype = torch.float32

                builder.add_parameter(node.name, list(shape), dtype)
                
            elif node.op == 'call_function':
                converter = OpRegistry.get_function(node.target)
                if converter:
                    try:
                        res = converter(builder, node, node.args, node.kwargs)
                        builder.register_output(node.name, res)
                    except Exception as e:
                        raise NPUCompilationError(f"Error converting function {node.target}: {e}") from e
                else:
                    raise NPUCompilationError(f"Function {node.target} not supported. Please file a feature request.")

            elif node.op == 'call_method':
                converter = OpRegistry.get_method(node.target)
                if converter:
                    try:
                        res = converter(builder, node, node.args, node.kwargs)
                        builder.register_output(node.name, res)
                    except Exception as e:
                        raise NPUCompilationError(f"Error converting method {node.target}: {e}") from e
                else:
                    raise NPUCompilationError(f"Method {node.target} not supported. Please file a feature request.")

            elif node.op == 'call_module':
                submod = model
                for atom in node.target.split('.'):
                    submod = getattr(submod, atom)
                
                converter = OpRegistry.get_module(type(submod))
                if converter:
                    try:
                        res = converter(builder, node, submod, node.args, node.kwargs)
                        builder.register_output(node.name, res)
                    except Exception as e:
                        raise NPUCompilationError(f"Error converting module {type(submod)}: {e}") from e
                else:
                    raise NPUCompilationError(f"Module type {type(submod)} not supported. Please file a feature request.")

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
        logger.info("Compiling model for NPU...")
        compiled = core.compile_model(ov_model, "NPU")
        
        # 5. Wrap
        return NPUGraphModule(compiled, [p.friendly_name for p in builder.parameters])
        
    except Exception as e:
        logger.error(f"Compilation Failed: {e}")
        raise e

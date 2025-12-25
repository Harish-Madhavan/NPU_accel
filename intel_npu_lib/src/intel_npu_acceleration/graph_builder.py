import torch
import torch.fx
from torch.fx.interpreter import Interpreter
import openvino as ov
import openvino.opset13 as ops
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ValueCapturingInterpreter(Interpreter):
    def __init__(self, module: torch.nn.Module, garbage_collect_values: bool = False):
        super().__init__(module, garbage_collect_values)
        self.node_values: Dict[torch.fx.Node, Any] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        val = super().run_node(n)
        self.node_values[n] = val
        return val

class OVGraphBuilder:
    def __init__(self, node_values: Optional[Dict[torch.fx.Node, Any]] = None):
        self.node_map: Dict[str, Any] = {} # Maps fx node name to OV node output
        self.parameters: List[Any] = []
        self.result_nodes: List[Any] = [] # List of OV nodes that are outputs
        self.node_values = node_values if node_values is not None else {}
        
    def get_input(self, node_name: str) -> Any:
        if node_name not in self.node_map:
            raise RuntimeError(f"Node {node_name} not found in graph map.")
        return self.node_map[node_name]

    def get_input_or_constant(self, arg: Any) -> Any:
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
            return [self.get_input_or_constant(x) for x in arg]
        elif arg is None:
            return None 
        else:
            raise NotImplementedError(f"Cannot convert argument type {type(arg)} to OV input")

    def add_parameter(self, name: str, shape: List[int], dtype: torch.dtype) -> Any:
        # Convert torch dtype to numpy dtype
        if dtype == torch.float32: np_type = np.float32
        elif dtype == torch.float16: np_type = np.float16
        elif dtype == torch.int64: np_type = np.int64
        elif dtype == torch.int8: np_type = np.int32 
        elif dtype == torch.uint8: np_type = np.int32
        elif dtype == torch.bool: np_type = bool
        else: np_type = np.float32 # Default
        
        param = ops.parameter(ov.Shape(shape), dtype=np_type, name=name)
        self.node_map[name] = param
        self.parameters.append(param)
        return param

    def add_constant(self, name: str, tensor: torch.Tensor) -> Any:
        # Tensor to numpy
        data = tensor.detach().cpu().numpy()
        try:
            const_node = ops.constant(data, name=name)
        except Exception as e:
            logger.error(f"Failed to create constant for {name}. Shape: {data.shape}, Dtype: {data.dtype}")
            raise e
        self.node_map[name] = const_node
        return const_node

    def register_output(self, name: str, ov_node: Any):
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

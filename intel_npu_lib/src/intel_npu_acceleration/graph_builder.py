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
        """Create an OV Parameter node and register it in the node_map."""
        _TORCH_TO_NP: Dict[torch.dtype, Any] = {
            torch.float32:  np.float32,
            torch.float16:  np.float16,
            torch.bfloat16: np.float32,   # OV represents bfloat16 separately; map to f32 for safety
            torch.float64:  np.float64,
            torch.int64:    np.int64,
            torch.int32:    np.int32,
            torch.int16:    np.int16,
            torch.int8:     np.int8,
            torch.uint8:    np.uint8,
            torch.bool:     bool,
        }
        np_type = _TORCH_TO_NP.get(dtype, np.float32)  # safe fallback

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
        """Promote inp0 and inp1 to a common OV element type.

        Promotion rules
        ---------------
        float × float  →  higher precision (f32 beats f16)
        float × int    →  float type
        int   × int    →  higher-width integer
                          (i8/u8 → i32, i16 → i32, i32 stays i32, i64 stays i64)
        """
        t0 = inp0.get_element_type()
        t1 = inp1.get_element_type()

        if t0 == t1:
            return inp0, inp1

        s0 = str(t0)
        s1 = str(t1)

        is_f0 = s0.startswith('f') or 'bf' in s0
        is_f1 = s1.startswith('f') or 'bf' in s1

        target_type: Any = None

        if is_f0 and is_f1:
            # Both float: pick the wider one
            if '64' in s0 or '64' in s1:
                target_type = np.float64
            elif '32' in s0 or '32' in s1:
                target_type = np.float32
            else:
                target_type = np.float16

        elif is_f0:
            # inp0 is float, inp1 is int → promote to inp0's float type
            if '64' in s0:   target_type = np.float64
            elif '32' in s0: target_type = np.float32
            else:            target_type = np.float16

        elif is_f1:
            # inp1 is float, inp0 is int → promote to inp1's float type
            if '64' in s1:   target_type = np.float64
            elif '32' in s1: target_type = np.float32
            else:            target_type = np.float16

        else:
            # Both integer: widen to the larger type.
            # OV element type strings: i8, u8, i16, i32, i64
            _INT_WIDTH = {'i8': 8, 'u8': 8, 'i16': 16, 'i32': 32, 'i64': 64}
            w0 = _INT_WIDTH.get(s0, 32)
            w1 = _INT_WIDTH.get(s1, 32)
            max_w = max(w0, w1)
            if max_w <= 8:
                target_type = np.int8
            elif max_w <= 16:
                target_type = np.int16
            elif max_w <= 32:
                target_type = np.int32
            else:
                target_type = np.int64

        if target_type is not None:
            if t0 != target_type:
                inp0 = ops.convert(inp0, destination_type=target_type)
            if t1 != target_type:
                inp1 = ops.convert(inp1, destination_type=target_type)

        return inp0, inp1

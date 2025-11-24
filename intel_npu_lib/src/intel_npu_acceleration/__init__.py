import torch
import importlib.util
import sys
import os
import platform
from typing import Optional

# On Windows, we need to explicitly add OpenVINO DLLs to the search path
if platform.system() == "Windows":
    try:
        import openvino
        libs_dir = os.path.join(os.path.dirname(openvino.__file__), "libs")
        if os.path.exists(libs_dir):
            os.add_dll_directory(libs_dir)
    except (ImportError, AttributeError):
        pass

try:
    from . import _C
except ImportError as e:
    import warnings
    warnings.warn(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
    _C = None

def is_available():
    """Checks if the Intel NPU is available on the system."""
    if _C is None:
        return False
    return _C.is_npu_available()

def add(a, b):
    if _C is not None: return _C.npu_add(a, b)
    return a + b

def sub(a, b):
    if _C is not None: return _C.npu_sub(a, b)
    return a - b

def mul(a, b):
    if _C is not None: return _C.npu_mul(a, b)
    return a * b

def div(a, b):
    if _C is not None: return _C.npu_div(a, b)
    return a / b

def matmul(a, b):
    if _C is not None: return _C.npu_matmul(a, b)
    return torch.matmul(a, b)

# --- Integration Layer ---

import torch.fx

class NPUCompiler:
    """
    A simple compiler that replaces supported torch operations with 
    Intel NPU accelerated operations in a torch.nn.Module.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.traced_model = torch.fx.symbolic_trace(model)

    def compile(self) -> torch.nn.Module:
        """
        Transforms the model to use NPU operations where possible.
        """
        if _C is None:
            print("Warning: Intel NPU extension not loaded. Returning original model.")
            return self.model

        # Define mapping from torch ops to npu ops
        # Note: We need to handle the operator functions (like operator.add) and torch functions
        import operator
        
        op_map = {
            torch.add: add,
            operator.add: add,
            torch.sub: sub,
            operator.sub: sub,
            torch.mul: mul,
            operator.mul: mul,
            torch.div: div,
            operator.truediv: div,
            torch.matmul: matmul,
        }

        for node in self.traced_model.graph.nodes:
            if node.op == 'call_function':
                if node.target in op_map:
                    print(f"Replacing {node.target} with NPU op")
                    node.target = op_map[node.target]
        
        self.traced_model.graph.lint()
        self.traced_model.recompile()
        return self.traced_model

def compile(model: torch.nn.Module, sample_input: Optional[torch.Tensor] = None) -> torch.nn.Module:
    """
    Compiles a PyTorch model to run on the Intel NPU.
    
    Args:
        model: The PyTorch model (nn.Module).
        sample_input: Optional sample input (not currently used by this simple compiler, 
                      but kept for API compatibility with other compilers).
    
    Returns:
        A compiled torch.nn.Module that dispatches to NPU operations.
    """
    compiler = NPUCompiler(model)
    return compiler.compile()
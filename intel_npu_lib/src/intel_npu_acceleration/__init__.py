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
    if _C is None: return False
    return _C.is_npu_available()

def add(a, b): return _C.npu_add(a, b) if _C else a + b
def sub(a, b): return _C.npu_sub(a, b) if _C else a - b
def mul(a, b): return _C.npu_mul(a, b) if _C else a * b
def div(a, b): return _C.npu_div(a, b) if _C else a / b
def matmul(a, b): return _C.npu_matmul(a, b) if _C else torch.matmul(a, b)
def relu(a): return _C.npu_relu(a) if _C else torch.relu(a)
def gelu(a): return _C.npu_gelu(a) if _C else torch.nn.functional.gelu(a)
def silu(a): return _C.npu_silu(a) if _C else torch.nn.functional.silu(a)
def softmax(a, dim): return _C.npu_softmax(a, dim) if _C else torch.softmax(a, dim)
def linear(input, weight, bias): return _C.npu_linear(input, weight, bias) if _C else torch.nn.functional.linear(input, weight, bias)
def rmsnorm(input, weight, eps): return _C.npu_rmsnorm(input, weight, eps) if _C else _torch_rmsnorm(input, weight, eps)

def transpose(a, dim0, dim1):
    if _C is None: return torch.transpose(a, dim0, dim1)
    # Construct permutation list
    rank = a.dim()
    perm = list(range(rank))
    # Handle negative dims
    if dim0 < 0: dim0 += rank
    if dim1 < 0: dim1 += rank
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return _C.npu_transpose(a, perm)

def reshape(a, *shape):
    if _C is None: return a.reshape(*shape)
    # Normalize shape: could be passed as tuple or varargs
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        target_shape = list(shape[0])
    else:
        target_shape = list(shape)
    
    # Handle -1 in shape (simple inference)
    # Note: C++ implementation might prefer explicit shape, but let's try passing it.
    # If -1 exists, we should calculate it to be safe, or let OpenVINO handle it if we mapped -1 to 0 or -1.
    # Current C++ uses special_zero=false, so 0 means 0. OpenVINO Reshape usually takes 0 to copy dim, -1 to infer.
    # Let's pass as is, assuming C++ handles it or OpenVINO does.
    # Actually, ops.cpp Reshape(..., false) means 0 is 0.
    # It's safer to resolve -1 here in Python if possible, but that requires knowing total elements.
    # Let's rely on PyTorch to infer shape for now? No, we want to avoid PyTorch compute.
    # Let's just pass it. If target_shape has -1, OpenVINO treats -1 as infer.
    return _C.npu_reshape(a, target_shape)

# Custom _torch_rmsnorm for CPU fallback, as it's not a standard torch.nn.functional
def _torch_rmsnorm(input, weight, eps):
    # This is a simplified RMSNorm, might need adjustment for full compatibility
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return input * weight

# Register custom op for FX tracing
torch.fx.wrap('rmsnorm')

# --- Integration Layer ---

import torch.fx
import operator

class NPUCompiler:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.traced_model = torch.fx.symbolic_trace(model)

    def compile(self) -> torch.nn.Module:
        if _C is None:
            print("Warning: Intel NPU extension not loaded. Returning original model.")
            return self.model

        # 1. Replace Function Calls
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
            torch.relu: relu,
            torch.nn.functional.relu: relu,
            torch.nn.functional.gelu: gelu,
            torch.nn.functional.silu: silu,
            torch.softmax: softmax,
            torch.nn.functional.softmax: softmax,
            torch.nn.functional.linear: linear,
            _torch_rmsnorm: rmsnorm, # Map our CPU fallback to NPU rmsnorm
            torch.transpose: transpose,
            torch.ops.aten.transpose.int: transpose,
            torch.reshape: reshape,
            torch.ops.aten.reshape: reshape,
            torch.ops.aten.view: reshape, # View is essentially reshape
        }

        for node in self.traced_model.graph.nodes:
            if node.op == 'call_function':
                if node.target in op_map:
                    print(f"Replacing function {node.target} with NPU op")
                    node.target = op_map[node.target]
            
            # 2. Replace Modules (Linear, ReLU, etc.)
            # FX doesn't automatically decompose modules into function calls unless explicitly told.
            # Instead, we can replace the module instance itself with a custom NPU module
            # or keep it as a call_module and rely on a custom Interpreter.
            # Simpler approach: Replace the module in the graph with a call to our functional NPU op
            # IF it's stateless (like ReLU). For Stateful (Linear), we need access to weights.
            
            if node.op == 'call_module':
                submodule = self.traced_model.get_submodule(node.target)
                if isinstance(submodule, torch.nn.Linear):
                    print(f"Replacing module {node.target} (Linear) with NPU Linear")
                    # We need to fetch weights/bias and inject them as inputs to functional linear
                    # This is tricky in FX without full graph rewriting.
                    # EASIER: Create a wrapper NPU Linear module.
                    pass 

        # Module Replacement Strategy for Stateful Modules (Linear)
        # We iterate over named modules and swap them.
        new_modules = {}
        for name, module in self.traced_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                new_modules[name] = NPULinear(module)
            elif isinstance(module, torch.nn.ReLU):
                new_modules[name] = NPUReLU()
            elif isinstance(module, torch.nn.GELU):
                new_modules[name] = NPUGELU()
            elif isinstance(module, torch.nn.Softmax):
                new_modules[name] = NPUSoftmax(module.dim)
            elif isinstance(module, torch.nn.SiLU):
                new_modules[name] = NPUSiLU()

        for name, new_module in new_modules.items():
            # traverse to parent to set attribute
            parent_name, _, child_name = name.rpartition('.')
            parent = self.traced_model.get_submodule(parent_name) if parent_name else self.traced_model
            setattr(parent, child_name, new_module)

        self.traced_model.graph.lint()
        self.traced_model.recompile()
        return self.traced_model

class NPULinear(torch.nn.Module):
    def __init__(self, original_linear: torch.nn.Linear):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
    
    def forward(self, input):
        # Handle optional bias
        b = self.bias if self.bias is not None else torch.zeros(self.weight.size(0))
        return linear(input, self.weight, b)

class NPUReLU(torch.nn.Module):
    def forward(self, input):
        return relu(input)

class NPUGELU(torch.nn.Module):
    def forward(self, input):
        return gelu(input)

class NPUSoftmax(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    def forward(self, input):
        return softmax(input, self.dim)

class NPUSiLU(torch.nn.Module):
    def forward(self, input):
        return silu(input)

class NPURMSNorm(torch.nn.Module):
    def __init__(self, original_rmsnorm_module):
        super().__init__()
        self.weight = original_rmsnorm_module.weight
        self.eps = original_rmsnorm_module.eps if original_rmsnorm_module.eps is not None else 1e-5
    
    def forward(self, input):
        return rmsnorm(input, self.weight, self.eps)

def compile(model: torch.nn.Module, sample_input: Optional[torch.Tensor] = None) -> torch.nn.Module:
    compiler = NPUCompiler(model)
    return compiler.compile()

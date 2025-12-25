import os
import sys
import platform
import logging

# --- Logging Setup ---
logger = logging.getLogger("intel_npu_acceleration")
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- DLL Loading (Windows) ---
if platform.system() == "Windows":
    try:
        import openvino
        libs_dir = os.path.join(os.path.dirname(openvino.__file__), "libs")
        if os.path.exists(libs_dir):
            os.add_dll_directory(libs_dir)
    except (ImportError, AttributeError):
        pass

# --- Import Core ---
try:
    from . import _C
except ImportError as e:
    logger.warning(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
    _C = None

def is_available() -> bool:
    if _C is None: return False
    return _C.is_npu_available()

# --- Expose Functional API ---
from .functional import (
    add, sub, mul, div, neg,
    matmul, linear,
    relu, gelu, silu, softmax,
    rmsnorm,
    transpose, reshape,
    update_kv_cache
)

# --- Expose Compiler API ---
from .frontend import compile_to_npu
compile = compile_to_npu

__all__ = [
    "is_available",
    "compile_to_npu",
    "add", "sub", "mul", "div", "neg",
    "matmul", "linear",
    "relu", "gelu", "silu", "softmax",
    "rmsnorm",
    "transpose", "reshape",
    "update_kv_cache"
]
import torch
import importlib.util
import sys
import os
import platform

# On Windows, we need to explicitly add OpenVINO DLLs to the search path
if platform.system() == "Windows":
    try:
        import openvino
        libs_dir = os.path.join(os.path.dirname(openvino.__file__), "libs")
        if os.path.exists(libs_dir):
            os.add_dll_directory(libs_dir)
    except (ImportError, AttributeError):
        # AttributeError can happen if openvino package structure changes or __file__ is missing
        pass

# Try to import the compiled C++ extension
try:
    from . import _C
except ImportError as e:
    # This happens if the package is not yet built/installed
    # We provide a warning or handling here
    import warnings
    warnings.warn(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}. Ensure the library is installed correctly.")
    _C = None

def is_available():
    """Checks if the Intel NPU is available on the system."""
    if _C is None:
        return False
    return _C.is_npu_available()

def add(a, b):
    """
    Performs element-wise addition using NPU acceleration (if available).
    Falls back to CPU or standard PyTorch dispatch if NPU is unavailable.
    """
    if _C is not None:
        return _C.npu_add(a, b)
    return a + b

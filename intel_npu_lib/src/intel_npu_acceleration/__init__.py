"""
Intel NPU Acceleration Library for PyTorch.
Hardware-accelerated deep learning execution on Intel Core Ultra Neural Processing Units.
"""

import logging
import sys

# --- Package Version (single-source: prefer installed distribution metadata) ---
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("intel_npu_acceleration")
except Exception:
    __version__ = "0.2.0"

# --- Logging Setup ---
# Respects NPU_LOG_LEVEL env var and never overrides an explicit app config.
logger = logging.getLogger("intel_npu_acceleration")
_handler = logging.StreamHandler(sys.stderr)
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_handler)
if not logger.level:
    import os as _os

    logger.setLevel(_os.environ.get("NPU_LOG_LEVEL", "INFO").upper())

# --- Windows DLL Path Loading (OpenVINO + PyTorch) ---
# Must be executed before importing any submodules so native _C can resolve DLLs.
from ._dll import setup_windows_dll_directories

setup_windows_dll_directories()

# --- Exceptions ---
# --- Neural Network Modules & Quantization ---
from . import nn, optim

# --- Disk & Memory Cache Management ---
from .cache import (  # noqa: E402
    clean_old_cache,
    clear_cache,
    ensure_cache_dir,
    get_cache_dir,
    get_cache_size,
    get_cache_version,
    set_cache_dir,
)

# --- Device Management & Seamless Acceleration ---
from .device import (  # noqa: E402
    _C,
    accelerate,
    current_device,
    device_count,
    empty_cache,
    enable_sda,
    enable_turbo,
    get_device_name,
    get_device_properties,
    get_eager_device,
    get_performance_hint,
    get_turbo_state,
    is_available,
    performance_mode,
    set_eager_device,
    set_performance_hint,
    set_property,
    synchronize,
    turbo,
)
from .exceptions import (  # noqa: E402
    NPUCompilationError,
    NPUDeviceError,
    NPUError,
    NPURuntimeError,
    NPUUnsupportedOpError,
)

# --- Graph Compiler & PyTorch 2.0 TorchDynamo Backend ---
from .frontend import (  # noqa: E402
    NPUDynamicGraphModule,
    NPUGraphModule,
    clear_graph_cache,
    compile,
    compile_to_npu,
    export_openvino_ir,
)

# --- Functional & Autograd Operations ---
from .functional import (  # noqa: E402
    abs,
    add,
    bce_with_logits_loss,
    cat,
    clamp,
    conv2d,
    cos,
    cross_entropy_loss,
    div,
    embedding,
    exp,
    flatten,
    full,
    gelu,
    hardsigmoid,
    hardswish,
    index_select,
    l1_loss,
    layer_norm,
    linear,
    matmul,
    max_pool2d,
    mean,
    mse_loss,
    mul,
    neg,
    ones,
    pow,
    quantized_linear,
    relu,
    reshape,
    rmsnorm,
    rotary_embedding,
    rsqrt,
    scaled_dot_product_attention,
    silu,
    sin,
    softmax,
    sqrt,
    squeeze,
    stack,
    sub,
    transpose,
    triu,
    unsqueeze,
    update_kv_cache,
    where,
    zeros,
)

# --- Diagnostic Introspection ---
from .info import get_system_info, print_info  # noqa: E402
from .nn import (  # noqa: E402
    NPUStatefulKVCache,
    quantize,
)
from .optim import NPUSGD, NPUAdam, clip_grad_norm_

# --- Register as proper PyTorch backend (torch.device("npu"), torch.npu, torch.backends.npu) ---
try:
    from .backends.npu_backend import register_pytorch_backend

    register_pytorch_backend()
except Exception as e:
    logger.debug(f"PyTorch NPU backend registration skipped: {e}")

__all__ = [
    # Metadata
    "__version__",
    # Exceptions
    "NPUError",
    "NPURuntimeError",
    "NPUDeviceError",
    "NPUCompilationError",
    "NPUUnsupportedOpError",
    # Device & Seamless Integration
    "is_available",
    "device_count",
    "current_device",
    "get_device_name",
    "get_device_properties",
    "empty_cache",
    "synchronize",
    "turbo",
    "performance_mode",
    "accelerate",
    "set_property",
    "set_performance_hint",
    "get_performance_hint",
    "enable_turbo",
    "get_turbo_state",
    "enable_sda",
    "set_eager_device",
    "get_eager_device",
    # Cache
    "get_cache_dir",
    "set_cache_dir",
    "ensure_cache_dir",
    "clear_cache",
    "get_cache_size",
    "get_cache_version",
    "clean_old_cache",
    "clear_graph_cache",
    # Diagnostics
    "get_system_info",
    "print_info",
    # Compiler
    "compile",
    "compile_to_npu",
    "export_openvino_ir",
    "NPUGraphModule",
    "NPUDynamicGraphModule",
    # NN Modules
    "NPUStatefulKVCache",
    "quantize",
    # Functional Ops
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "where",
    "neg",
    "matmul",
    "linear",
    "relu",
    "gelu",
    "silu",
    "sin",
    "cos",
    "exp",
    "sqrt",
    "abs",
    "rsqrt",
    "clamp",
    "triu",
    "flatten",
    "softmax",
    "rmsnorm",
    "transpose",
    "reshape",
    "squeeze",
    "unsqueeze",
    "cat",
    "stack",
    "mean",
    "index_select",
    "zeros",
    "ones",
    "full",
    "conv2d",
    "max_pool2d",
    "update_kv_cache",
    "rotary_embedding",
    "quantized_linear",
    "layer_norm",
    "hardsigmoid",
    "hardswish",
    "embedding",
    "scaled_dot_product_attention",
    "cross_entropy_loss",
    "l1_loss",
    "bce_with_logits_loss",
    "clip_grad_norm_",
]

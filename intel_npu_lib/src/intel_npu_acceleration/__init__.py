"""
Intel NPU Acceleration Library for PyTorch.
Hardware-accelerated deep learning execution on Intel Core Ultra Neural Processing Units.
"""

import sys
import logging

# --- Package Version ---
__version__ = "0.2.0"

# --- Logging Setup ---
logger = logging.getLogger("intel_npu_acceleration")
_handler = logging.StreamHandler(sys.stderr)
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# --- Exceptions ---
from .exceptions import (  # noqa: E402
    NPUError,
    NPURuntimeError,
    NPUDeviceError,
    NPUCompilationError,
    NPUUnsupportedOpError,
)

# --- Device Management & Properties ---
from .device import (  # noqa: E402
    _C,
    is_available,
    set_property,
    set_performance_hint,
    enable_turbo,
    enable_sda,
    set_eager_device,
    get_eager_device,
)

# --- Disk & Memory Cache Management ---
from .cache import (  # noqa: E402
    get_cache_dir,
    set_cache_dir,
    clear_cache,
    get_cache_size,
    get_cache_version,
    clean_old_cache,
)

# --- Neural Network Modules & Quantization ---
from .nn import (  # noqa: E402
    NPUStatefulKVCache,
    quantize,
)

# --- Diagnostic Introspection ---
from .info import get_system_info, print_info  # noqa: E402

# --- Graph Compiler & PyTorch 2.0 TorchDynamo Backend ---
from .frontend import (  # noqa: E402
    compile,
    compile_to_npu,
    export_openvino_ir,
    clear_graph_cache,
    NPUGraphModule,
    NPUDynamicGraphModule,
)

# --- Functional & Autograd Operations ---
from .functional import (  # noqa: E402
    add,
    sub,
    mul,
    div,
    neg,
    matmul,
    linear,
    relu,
    gelu,
    silu,
    softmax,
    rmsnorm,
    transpose,
    reshape,
    squeeze,
    unsqueeze,
    cat,
    stack,
    mean,
    index_select,
    zeros,
    ones,
    full,
    conv2d,
    max_pool2d,
    update_kv_cache,
    quantized_linear,
    layer_norm,
    hardsigmoid,
    hardswish,
    embedding,
    scaled_dot_product_attention,
)

__all__ = [
    # Metadata
    "__version__",
    # Exceptions
    "NPUError",
    "NPURuntimeError",
    "NPUDeviceError",
    "NPUCompilationError",
    "NPUUnsupportedOpError",
    # Device
    "is_available",
    "set_property",
    "set_performance_hint",
    "enable_turbo",
    "enable_sda",
    "set_eager_device",
    "get_eager_device",
    # Cache
    "get_cache_dir",
    "set_cache_dir",
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
    "neg",
    "matmul",
    "linear",
    "relu",
    "gelu",
    "silu",
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
    "quantized_linear",
    "layer_norm",
    "hardsigmoid",
    "hardswish",
    "embedding",
    "scaled_dot_product_attention",
]

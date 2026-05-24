import os
import sys
import platform
import logging

# --- Logging Setup ---
logger = logging.getLogger("intel_npu_acceleration")
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)

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
_CACHE_DIR = None

# --- Cache Initialization ---
if _C is not None:
    try:
        # Default cache location: npu_cache in the same directory as this file's parent or current working dir
        possible_cache_dirs = [
            os.path.join(os.getcwd(), "npu_cache"),
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "npu_cache")
            ),
        ]

        cache_dir = None
        for d in possible_cache_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                cache_dir = d
                break

        if cache_dir:
            _CACHE_DIR = cache_dir
            logger.info(f"Setting NPU cache directory: {cache_dir}")
            _C.set_cache_dir(cache_dir)
        else:
            # Optionally create it in CWD if not found
            cwd_cache = os.path.join(os.getcwd(), "npu_cache")
            if not os.path.exists(cwd_cache):
                try:
                    os.makedirs(cwd_cache, exist_ok=True)
                    _CACHE_DIR = cwd_cache
                    logger.info(f"Created NPU cache directory: {cwd_cache}")
                    _C.set_cache_dir(cwd_cache)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Failed to initialize disk cache: {e}")


def get_cache_dir():
    return _CACHE_DIR


def set_cache_dir(cache_dir: str):
    global _CACHE_DIR
    _CACHE_DIR = cache_dir
    logger.info(f"Setting NPU cache directory: {cache_dir}")
    if _C is not None:
        _C.set_cache_dir(cache_dir)


def is_available() -> bool:
    if _C is None:
        return False
    return _C.is_npu_available()


def set_property(key: str, value: str):
    """Set global NPU property (Level Zero backend)."""
    if _C is not None:
        _C.set_property(key, value)
    else:
        logger.warning("NPU C++ extension not loaded. Cannot set property.")


def set_performance_hint(hint: str):
    """Set global NPU performance hint (LATENCY, THROUGHPUT)."""
    if _C is not None:
        _C.set_performance_hint(hint)
    else:
        logger.warning("NPU C++ extension not loaded. Cannot set performance hint.")


def enable_turbo():
    """Enable Level Zero Turbo mode for maximum performance."""
    set_property("NPU_TURBO", "YES")


def enable_sda():
    """Enable Shared Device Address for zero-copy memory transfers."""
    set_property("NPU_USE_SDA", "YES")


# --- Expose Functional API ---
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
    conv2d,
    max_pool2d,
    update_kv_cache,
    quantized_linear,
)

# --- Expose Compiler API ---
from .frontend import compile, compile_to_npu  # noqa: E402

__all__ = [
    "is_available",
    "set_property",
    "set_performance_hint",
    "enable_turbo",
    "enable_sda",
    "compile",
    "compile_to_npu",
    "get_cache_dir",
    "set_cache_dir",
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
    "conv2d",
    "max_pool2d",
    "update_kv_cache",
    "quantized_linear",
]

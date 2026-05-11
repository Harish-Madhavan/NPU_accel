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
_C = None
try:
    from . import _C
except ImportError as e:
    logger.warning(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
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


def is_available() -> bool:
    if _C is None:
        return False
    return _C.is_npu_available()


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
)

# --- Expose Compiler API ---
from .frontend import compile_to_npu  # noqa: E402

compile = compile_to_npu

__all__ = [
    "is_available",
    "compile_to_npu",
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
]

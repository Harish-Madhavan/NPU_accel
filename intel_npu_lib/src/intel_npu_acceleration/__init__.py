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


def clear_cache():
    """Clear all files in the NPU cache directory."""
    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.exists(cache_dir):
        logger.warning("No cache directory configured or directory does not exist.")
        return
    logger.info(f"Clearing NPU cache directory: {cache_dir}")
    for root, dirs, files in os.walk(cache_dir, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except Exception as e:
                logger.debug(f"Failed to remove cache file {name}: {e}")
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception as e:
                logger.debug(f"Failed to remove cache directory {name}: {e}")


def get_cache_size() -> tuple[int, int]:
    """
    Get the total size in bytes and number of files in the cache directory.
    Returns:
        (total_size_bytes, file_count)
    """
    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.exists(cache_dir):
        return 0, 0
    total_size = 0
    file_count = 0
    for root, _, files in os.walk(cache_dir):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total_size += os.path.getsize(file_path)
                file_count += 1
            except Exception:
                pass
    return total_size, file_count


def clean_old_cache(max_size_mb: int = 1024, max_files: int = 500):
    """
    Clean the cache directory by deleting the oldest files (based on modification time)
    until the total size is below max_size_mb and total file count is below max_files.
    """
    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.exists(cache_dir):
        return

    # Gather all cache files with their modification times and sizes
    all_files = []
    for root, _, files in os.walk(cache_dir):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(file_path)
                size = os.path.getsize(file_path)
                all_files.append((file_path, mtime, size))
            except Exception:
                pass

    total_size = sum(f[2] for f in all_files)
    total_files = len(all_files)

    max_size_bytes = max_size_mb * 1024 * 1024

    if total_size <= max_size_bytes and total_files <= max_files:
        return

    # Sort files by modification time (oldest first)
    all_files.sort(key=lambda x: x[1])

    logger.info(
        f"Cleaning NPU cache. Current size: {total_size / (1024*1024):.2f} MB ({total_files} files). "
        f"Limits: {max_size_mb} MB, {max_files} files."
    )

    deleted_count = 0
    deleted_size = 0

    for file_path, _, size in all_files:
        if total_size <= max_size_bytes and total_files <= max_files:
            break
        try:
            os.remove(file_path)
            total_size -= size
            total_files -= 1
            deleted_count += 1
            deleted_size += size
        except Exception as e:
            logger.debug(f"Failed to delete old cache file {file_path}: {e}")

    logger.info(f"Cleaned {deleted_count} old cache files ({deleted_size / (1024*1024):.2f} MB cleared).")


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


def set_eager_device(device: str):
    """Set device for eager operations (CPU, NPU)."""
    if _C is not None:
        _C.set_eager_device(device)
    else:
        logger.warning("NPU C++ extension not loaded. Cannot set eager device.")


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
    "clear_cache",
    "get_cache_size",
    "clean_old_cache",
    "set_eager_device",
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

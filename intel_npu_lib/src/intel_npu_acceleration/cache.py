"""
Intel NPU Binary Model & Graph Disk Cache Management.

This module manages the persistent disk cache for pre-compiled OpenVINO intermediate
representation (IR) binary blobs and Level Zero hardware executables. Disk caching
eliminates model compilation overhead during subsequent application startups.
"""

import os
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger("intel_npu_acceleration.cache")

_CACHE_DIR: Optional[str] = None
_LAST_CLEANUP_TIME: float = 0.0


def _init_default_cache() -> None:
    """Initialize the default cache directory in the workspace or repository root."""
    global _CACHE_DIR
    try:
        from .device import _C
    except ImportError:
        _C = None

    try:
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
            if _C is not None and hasattr(_C, "set_cache_dir"):
                _C.set_cache_dir(cache_dir)
        else:
            cwd_cache = os.path.join(os.getcwd(), "npu_cache")
            if not os.path.exists(cwd_cache):
                try:
                    os.makedirs(cwd_cache, exist_ok=True)
                    _CACHE_DIR = cwd_cache
                    logger.info(f"Created NPU cache directory: {cwd_cache}")
                    if _C is not None and hasattr(_C, "set_cache_dir"):
                        _C.set_cache_dir(cwd_cache)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Failed to initialize disk cache: {e}")


# Initialize default disk cache on import
_init_default_cache()


def get_cache_dir() -> Optional[str]:
    """Get the absolute filesystem path of the current NPU model cache directory.

    Returns:
        Optional[str]: Path to the active cache directory, or None if unconfigured.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> print(npu.get_cache_dir())
    """
    return _CACHE_DIR


def set_cache_dir(cache_dir: str) -> None:
    """Configure a custom directory for persistent NPU binary model caching.

    Updates both the Python runtime and the native C++ Level Zero / OpenVINO cache director.

    Args:
        cache_dir (str): Target filesystem directory path for storing compiled binary blobs.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.set_cache_dir("/path/to/my_npu_cache")
    """
    global _CACHE_DIR
    _CACHE_DIR = cache_dir
    logger.info(f"Setting NPU cache directory: {cache_dir}")
    try:
        from .device import _C

        if _C is not None and hasattr(_C, "set_cache_dir"):
            _C.set_cache_dir(cache_dir)
    except ImportError:
        pass


def clear_cache() -> None:
    """Purge all cached compiled model binary blobs from the disk cache directory.

    Safely iterates through the cache directory and deletes all cached OpenVINO IR
    and Level Zero binary artifacts.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.clear_cache()
    """
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


def get_cache_size() -> Tuple[int, int]:
    """Calculate the total size and file count of the NPU disk cache.

    Returns:
        Tuple[int, int]: A tuple of `(total_size_bytes, file_count)`.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> size_bytes, num_files = npu.get_cache_size()
        >>> print(f"Cache holds {num_files} models ({size_bytes / (1024 * 1024):.2f} MB)")
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


def get_cache_version() -> int:
    """Retrieve the in-memory cache version counter.

    Increments whenever the in-memory C++ compiled model cache is mutated or cleared.

    Returns:
        int: The monotonic cache version counter.
    """
    try:
        from .device import _C

        if _C is not None and hasattr(_C, "get_cache_version"):
            return _C.get_cache_version()
    except ImportError:
        pass
    return 0


def clean_old_cache(max_size_mb: int = 2048, max_files: int = 10000) -> None:
    """Prune the oldest cached models (LRU based on modification time).

    Ensures the disk cache does not exceed specified capacity limits. Throttled to execute
    at most once every 60 seconds to avoid filesystem overhead during tight training loops.

    Args:
        max_size_mb (int, optional): Maximum allowed disk cache size in megabytes. Defaults to 2048.
        max_files (int, optional): Maximum allowed file count in cache directory. Defaults to 10000.
    """
    global _LAST_CLEANUP_TIME

    current_time = time.time()
    if current_time - _LAST_CLEANUP_TIME < 60.0:
        return

    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.exists(cache_dir):
        return

    _LAST_CLEANUP_TIME = current_time

    try:
        files = []
        for root, _, filenames in os.walk(cache_dir):
            for name in filenames:
                file_path = os.path.join(root, name)
                try:
                    stat = os.stat(file_path)
                    files.append((file_path, stat.st_mtime, stat.st_size))
                except Exception:
                    pass

        total_size = sum(f[2] for f in files)
        max_bytes = max_size_mb * 1024 * 1024

        if total_size > max_bytes or len(files) > max_files:
            files.sort(key=lambda x: x[1])
            for file_path, _, size in files:
                if total_size <= max_bytes and len(files) <= max_files:
                    break
                try:
                    os.remove(file_path)
                    total_size -= size
                    files.pop(0)
                except Exception as e:
                    logger.debug(
                        f"Failed to remove cache file {os.path.basename(file_path)}: {e}"
                    )
    except Exception as e:
        logger.debug(f"Error during cache cleanup: {e}")

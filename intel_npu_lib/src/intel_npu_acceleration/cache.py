"""
Intel NPU Binary Model & Graph Disk Cache Management.
"""

import os
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger("intel_npu_acceleration.cache")

_CACHE_DIR: Optional[str] = None
_LAST_CLEANUP_TIME: float = 0.0


def _init_default_cache():
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


# Initialize default disk cache
_init_default_cache()


def get_cache_dir() -> Optional[str]:
    """Get the current NPU model cache directory path."""
    return _CACHE_DIR


def set_cache_dir(cache_dir: str) -> None:
    """Set the NPU model cache directory path."""
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
    """Clear all compiled binary blob files in the NPU cache directory."""
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
    """
    Get total size in bytes and number of files in the cache directory.
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


def get_cache_version() -> int:
    """Get the current model cache version counter."""
    try:
        from .device import _C

        if _C is not None and hasattr(_C, "get_cache_version"):
            return _C.get_cache_version()
    except ImportError:
        pass
    return 0


def clean_old_cache(max_size_mb: int = 2048, max_files: int = 10000) -> None:
    """
    Clean cache directory by deleting the oldest files (LRU based on mtime)
    until size is below max_size_mb and file count is below max_files.
    Throttled to run at most once every 60 seconds.
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
            # Sort files by modification time (oldest first)
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

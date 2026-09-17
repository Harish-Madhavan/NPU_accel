"""
Intel NPU Binary Model & Graph Disk Cache Management.

This module manages the persistent disk cache for pre-compiled OpenVINO intermediate
representation (IR) binary blobs and Level Zero hardware executables. Disk caching
eliminates model compilation overhead during subsequent application startups.
"""

__all__ = [
    "get_cache_dir",
    "set_cache_dir",
    "ensure_cache_dir",
    "clear_cache",
    "get_cache_size",
    "get_cache_version",
    "clean_old_cache",
]

import logging
import os
import threading
import time

logger = logging.getLogger("intel_npu_acceleration.cache")

_CACHE_DIR: str | None = None
_LAST_CLEANUP_TIME: float = 0.0
_CACHE_LOCK = threading.Lock()


def _init_default_cache() -> None:
    """Initialize the default cache directory if one already exists.

    Priority: ``INTEL_NPU_CACHE_DIR`` / ``NPU_CACHE_DIR`` env vars first
    (created on demand), then an existing ``./npu_cache`` next to the
    current working directory or the repo checkout.

    Deliberately side-effect free apart from the explicit env-var case:
    never creates directories on import unless the user explicitly asked
    for a location via environment. Directory creation otherwise happens
    lazily in `set_cache_dir` / `ensure_cache_dir`.
    """
    global _CACHE_DIR
    try:
        from .device import _C
    except ImportError:
        _C = None  # noqa: N806

    try:
        # Explicit user request always wins (docs promise this variable).
        for env_var in ("INTEL_NPU_CACHE_DIR", "NPU_CACHE_DIR"):
            env_dir = os.environ.get(env_var)
            if env_dir:
                target = os.path.abspath(os.path.expandvars(os.path.expanduser(env_dir)))
                try:
                    os.makedirs(target, exist_ok=True)
                except Exception as e:
                    logger.debug(f"Failed to create env cache dir '{target}': {e}")
                    continue
                _CACHE_DIR = target
                logger.info(f"Setting NPU cache directory from {env_var}: {target}")
                if _C is not None and hasattr(_C, "set_cache_dir"):
                    _C.set_cache_dir(target)
                return

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
    except Exception as e:
        logger.debug(f"Failed to initialize disk cache: {e}")


# Initialize default disk cache on import
_init_default_cache()


def ensure_cache_dir(cache_dir: str | None = None) -> str | None:
    """Ensure the cache directory exists, creating it on demand.

    Args:
        cache_dir: Directory to ensure. Defaults to the configured cache dir.

    Returns:
        The ensured directory path, or None if unconfigured.
    """
    global _CACHE_DIR
    target = cache_dir or _CACHE_DIR
    if not target:
        return None
    try:
        os.makedirs(target, exist_ok=True)
        _CACHE_DIR = target
        return target
    except Exception as e:
        logger.debug(f"Failed to create cache directory '{target}': {e}")
        return None


def get_cache_dir() -> str | None:
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
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        logger.debug(f"Failed to create cache directory '{cache_dir}': {e}")
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


def get_cache_size() -> tuple[int, int]:
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
    at most once per `NPU_CACHE_THROTTLE_SEC` (see `config.cache_throttle_seconds`)
    to avoid filesystem overhead during tight training loops.

    Args:
        max_size_mb (int, optional): Maximum allowed disk cache size in megabytes. Defaults to 2048.
        max_files (int, optional): Maximum allowed file count in cache directory. Defaults to 10000.
    """
    global _LAST_CLEANUP_TIME

    from .config import config as npu_config

    with _CACHE_LOCK:
        current_time = time.time()
        if current_time - _LAST_CLEANUP_TIME < npu_config.cache_throttle_seconds:
            return

        cache_dir = get_cache_dir()
        if not cache_dir or not os.path.exists(cache_dir):
            return

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

            if total_size <= max_bytes and len(files) <= max_files:
                _LAST_CLEANUP_TIME = current_time
                return

            files.sort(key=lambda x: x[1])

            removed = 0
            for file_path, _, size in list(files):
                if total_size <= max_bytes and (len(files) - removed) <= max_files:
                    break
                try:
                    os.remove(file_path)
                    total_size -= size
                    removed += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to remove cache file {os.path.basename(file_path)}: {e}"
                    )

            for root, dirs, _ in os.walk(cache_dir, topdown=False):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass

            _LAST_CLEANUP_TIME = current_time
        except Exception as e:
            logger.debug(f"Error during cache cleanup: {e}")
            return

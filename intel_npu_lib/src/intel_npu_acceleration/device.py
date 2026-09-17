"""
Intel NPU Device Management, Discovery, and Runtime Properties Configuration.

This module provides hardware discovery, runtime property querying, power/performance
hint configuration, and execution target management for Intel® Neural Processing Units (NPUs)
via oneAPI Level Zero and the OpenVINO™ runtime backend.
"""

__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "get_device_name",
    "get_device_properties",
    "empty_cache",
    "synchronize",
    "set_property",
    "set_performance_hint",
    "get_performance_hint",
    "enable_turbo",
    "get_turbo_state",
    "enable_sda",
    "set_eager_device",
    "get_eager_device",
    "turbo",
    "performance_mode",
    "accelerate",
]

import contextlib
import logging
import threading
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("intel_npu_acceleration.device")

_TURBO_LOCK = threading.Lock()
_PERF_HINT_LOCK = threading.Lock()
_CURRENT_PERF_HINT: str = "LATENCY"
_CURRENT_TURBO: bool | None = None

_OV_CORE_LOCK = threading.Lock()
_OV_CORE: Any | None = None


def _get_ov_core() -> Any | None:
    """Return a shared OpenVINO Core instance (cached to avoid re-probing)."""
    global _OV_CORE
    if _OV_CORE is not None:
        return _OV_CORE
    with _OV_CORE_LOCK:
        if _OV_CORE is not None:
            return _OV_CORE
        try:
            import openvino as ov

            _OV_CORE = ov.Core()
        except Exception as e:
            logger.debug(f"OpenVINO Core init failed: {e}")
            _OV_CORE = None
        return _OV_CORE

# --- Windows DLL Path Loading (OpenVINO + PyTorch) ---
from ._dll import setup_windows_dll_directories

setup_windows_dll_directories()

# --- Native C++ Extension Loading ---
try:
    from . import _C
except ImportError as e:
    logger.warning(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
    _C = None

_EAGER_DEVICE: str = "NPU"


def _apply_to_ov_core(properties: dict[str, str]) -> None:
    """Mirror driver properties to the Python OpenVINO Core singleton.

    Graph-mode compiles use the Python ``ov.Core`` (not the C++ backend's
    Core), so global settings must reach both. Best-effort: the NPU plugin
    accepts unknown keys silently and per-compile configs override globals.
    """
    try:
        from .frontend.compiler import _get_core

        core = _get_core()
        core.set_property("NPU", properties)
    except Exception as e:
        logger.debug(f"OV Core property mirror skipped: {e}")


def is_available() -> bool:
    """Check if an Intel NPU device is present, initialized, and accessible.

    Probes the underlying Level Zero driver and OpenVINO runtime to verify that an
    Intel NPU accelerator is physically present and ready for hardware execution.

    Returns:
        bool: True if an Intel NPU is detected and functional, False otherwise.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> if npu.is_available():
        ...     print("Intel NPU is ready for hardware acceleration!")
    """
    if _C is not None:
        try:
            return _C.is_npu_available()
        except Exception as e:
            logger.debug(f"C++ NPU probe failed: {e}")

    core = _get_ov_core()
    if core is None:
        return False
    try:
        return "NPU" in core.available_devices
    except Exception as e:
        logger.debug(f"OpenVINO NPU probe failed: {e}")
        return False


def set_property(key: str, value: str) -> None:
    """Set low-level device configuration properties on the active NPU device.

    Passes configuration parameters directly to the Intel oneAPI Level Zero driver
    and OpenVINO NPU plugin.

    Args:
        key (str): The configuration property key (e.g., 'NPU_TURBO', 'NPU_USE_SDA').
        value (str): The property value (e.g., 'YES', 'NO', 'LATENCY').

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.set_property("NPU_TURBO", "YES")
    """
    if _C is not None and hasattr(_C, "set_property"):
        try:
            _C.set_property(key, value)
        except Exception as e:
            logger.warning(f"Failed to set property '{key}'='{value}': {e}")
    # Graph mode uses the Python ov.Core singleton (separate from the C++
    # backend's Core), so mirror there as well.
    _apply_to_ov_core({key: value})


def set_performance_hint(hint: str) -> None:
    """Configure runtime performance hint for compiled NPU executions."""
    global _CURRENT_PERF_HINT
    valid_hints = {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"}
    hint_upper = hint.upper()
    if hint_upper not in valid_hints:
        raise ValueError(f"Invalid performance hint '{hint}'. Expected one of: {valid_hints}")
    with _PERF_HINT_LOCK:
        _CURRENT_PERF_HINT = hint_upper
    if _C is not None and hasattr(_C, "set_performance_hint"):
        try:
            _C.set_performance_hint(hint_upper)
        except Exception as e:
            logger.warning(f"Failed to set performance hint '{hint_upper}': {e}")
    _apply_to_ov_core({"PERFORMANCE_HINT": hint_upper})


def get_performance_hint() -> str:
    """Return the currently configured performance hint."""
    with _PERF_HINT_LOCK:
        return _CURRENT_PERF_HINT


def enable_turbo(enable: bool = True) -> None:
    """Enable or disable Intel NPU Turbo execution mode for burst workloads."""
    global _CURRENT_TURBO
    with _TURBO_LOCK:
        _CURRENT_TURBO = bool(enable)
    if _C is not None and hasattr(_C, "enable_turbo"):
        try:
            _C.enable_turbo(enable)
        except Exception as e:
            logger.warning(f"Failed to set turbo mode to {enable}: {e}")
    _apply_to_ov_core({"NPU_TURBO": "YES" if enable else "NO"})


def get_turbo_state() -> bool | None:
    """Return the last requested turbo state, or None if never set."""
    with _TURBO_LOCK:
        return _CURRENT_TURBO


def enable_sda(enable: bool = True) -> None:
    """Enable or disable Intel Smart Dispatch Acceleration (SDA).

    Controls whether driver-level smart dispatch is enabled to minimize command queue
    submission latencies.

    Args:
        enable (bool, optional): True to enable SDA, False to disable. Defaults to True.
    """
    if _C is not None and hasattr(_C, "enable_sda"):
        try:
            _C.enable_sda(enable)
        except Exception as e:
            logger.warning(f"Failed to set SDA mode to {enable}: {e}")
    _apply_to_ov_core({"NPU_USE_SDA": "YES" if enable else "NO"})


def set_eager_device(device_name: str) -> None:
    """Set the execution device for eager single-op executions.

    Args:
        device_name (str): The target execution device, typically 'NPU' or 'CPU'.

    Raises:
        ValueError: If device_name is not 'NPU' or 'CPU'.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.set_eager_device("NPU")
    """
    global _EAGER_DEVICE
    device_upper = device_name.upper()
    if device_upper not in {"NPU", "CPU"}:
        raise ValueError(f"Invalid eager device '{device_name}'. Expected 'NPU' or 'CPU'.")
    _EAGER_DEVICE = device_upper
    if _C is not None and hasattr(_C, "set_eager_device"):
        try:
            _C.set_eager_device(_EAGER_DEVICE)
        except Exception as e:
            logger.debug(f"Failed to set C++ eager device: {e}")


def get_eager_device() -> str:
    """Get the currently configured execution device for eager operations.

    Returns:
        str: 'NPU' or 'CPU'.
    """
    return _EAGER_DEVICE


def device_count() -> int:
    """Return the number of Intel NPU devices available.

    Returns:
        int: 1 if Intel NPU is available and accessible, 0 otherwise.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> count = npu.device_count()
    """
    return 1 if is_available() else 0


def current_device() -> int:
    """Return the index of the currently selected NPU device.

    Returns:
        int: Device index (always 0 for primary NPU accelerator).
    """
    return 0


def get_device_name(device: int = 0) -> str:
    """Get the model / hardware brand name of the Intel NPU accelerator.

    Args:
        device (int, optional): Device index. Defaults to 0.

    Returns:
        str: Hardware description (e.g., 'Intel(R) AI Boost' or 'Intel NPU').

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> name = npu.get_device_name()
    """
    if not is_available():
        return "No Intel NPU detected"
    core = _get_ov_core()
    if core is None:
        return "Intel(R) AI Boost (NPU)"
    try:
        return core.get_property("NPU", "FULL_DEVICE_NAME")
    except Exception:
        return "Intel(R) AI Boost (NPU)"


def get_device_properties(device: int = 0) -> dict[str, Any]:
    """Get device configuration and hardware capability metadata.

    Args:
        device (int, optional): Device index. Defaults to 0.

    Returns:
        Dict[str, Any]: Dictionary containing device name, optimal streams, and supported properties.
    """
    available = is_available()
    props = {
        "name": get_device_name(device) if available else "No Intel NPU detected",
        "device_id": device,
        "is_available": available,
    }
    if available:
        core = _get_ov_core()
        try:
            props["optimal_infer_requests"] = core.get_property("NPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        except Exception:
            props["optimal_infer_requests"] = 1
    return props


def empty_cache() -> None:
    """Release all cached compiled model graph instances from memory and disk.

    Equivalent to `torch.cuda.empty_cache()` for Intel NPU acceleration.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.empty_cache()
    """
    from .cache import clear_cache
    from .frontend.compiler import clear_graph_cache

    clear_graph_cache()
    clear_cache()
    logger.info("NPU memory and model caches purged successfully.")


def synchronize(device: int | None = None) -> None:
    """Wait for all outstanding asynchronous operations on the NPU to complete.

    Equivalent to `torch.cuda.synchronize()` for Intel NPU acceleration.

    Args:
        device (Optional[int], optional): Device index to synchronize. Defaults to None.
    """
    # OpenVINO / Level Zero handles synchronous fences on each InferRequest::infer() call
    pass


@contextlib.contextmanager
def turbo(enable: bool = True) -> Iterator[None]:
    """Context manager to temporarily enable or disable Intel NPU Turbo boost mode."""
    prev = get_turbo_state()
    restore_to: bool | None = prev
    try:
        enable_turbo(enable)
        yield
    finally:
        if restore_to is not None:
            enable_turbo(restore_to)
        else:
            enable_turbo(False)


@contextlib.contextmanager
def performance_mode(hint: str = "LATENCY") -> Iterator[None]:
    """Context manager to temporarily set NPU performance mode."""
    prev = get_performance_hint()
    try:
        set_performance_hint(hint)
        yield
    finally:
        set_performance_hint(prev)


def accelerate(model: Any, example_input: Any = None, **kwargs: Any) -> Any:
    """One-line seamless accelerator function for PyTorch models.

    Automatically compiles and optimizes standard PyTorch `nn.Module` objects for Intel NPU.

    Args:
        model: PyTorch model or callable to accelerate.
        example_input (optional): Example tensor for static graph compilation. If omitted, uses PyTorch 2.x `torch.compile(backend="npu")`.
        **kwargs: Additional compiler flags forwarded to `torch.compile` or `compile_to_npu`.

    Returns:
        Accelerated NPU model instance.

    Examples:
        >>> import torch
        >>> import intel_npu_acceleration as npu
        >>> model = torch.nn.Linear(10, 2)
        >>> npu_model = npu.accelerate(model)
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        return model

    if example_input is not None:
        from .frontend.compiler import compile_to_npu
        return compile_to_npu(model, example_input, **kwargs)

    return torch.compile(model, backend="npu", **kwargs)


# --- PyTorch nn.Module Seamless Integration Extensions ---
try:
    import torch.nn as nn

    def _module_to_npu(self: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
        """Convenience method to prepare module for NPU execution."""
        return self

    def _module_compile_npu(
        self: nn.Module, example_input: Any = None, **kwargs: Any
    ) -> nn.Module:
        """Compile this module for Intel NPU hardware acceleration."""
        return accelerate(self, example_input=example_input, **kwargs)

    if not hasattr(nn.Module, "to_npu"):
        nn.Module.to_npu = _module_to_npu  # type: ignore[attr-defined]
    if not hasattr(nn.Module, "compile_npu"):
        nn.Module.compile_npu = _module_compile_npu  # type: ignore[attr-defined]

except (ImportError, AttributeError):
    pass


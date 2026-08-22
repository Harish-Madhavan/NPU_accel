"""
Intel NPU Device Management, Discovery, and Runtime Properties Configuration.

This module provides hardware discovery, runtime property querying, power/performance
hint configuration, and execution target management for Intel® Neural Processing Units (NPUs)
via oneAPI Level Zero and the OpenVINO™ runtime backend.
"""

import os
import platform
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("intel_npu_acceleration.device")

# --- Windows DLL Path Loading (OpenVINO + PyTorch) ---
if platform.system() == "Windows":
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
    except (ImportError, AttributeError):
        pass

    try:
        import openvino

        ov_dir = os.path.dirname(openvino.__file__)
        for d in [ov_dir, os.path.join(ov_dir, "libs"), os.path.join(ov_dir, "lib")]:
            if os.path.exists(d):
                os.add_dll_directory(d)
    except (ImportError, AttributeError):
        pass

# --- Native C++ Extension Loading ---
try:
    from . import _C
except ImportError as e:
    logger.warning(f"Could not load C++ extension 'intel_npu_acceleration._C': {e}")
    _C = None

_EAGER_DEVICE: str = "NPU"


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

    try:
        import openvino as ov

        core = ov.Core()
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


def set_performance_hint(hint: str) -> None:
    """Configure runtime performance hint for compiled NPU executions.

    Configures whether the hardware scheduler prioritizes minimum single-request latency
    or maximum parallel throughput.

    Args:
        hint (str): Performance mode. Must be one of:
            - 'LATENCY': Minimizes time-to-first-token and per-inference execution latency.
            - 'THROUGHPUT': Maximizes inferences per second by utilizing multiple hardware streams.
            - 'CUMULATIVE_THROUGHPUT': Aggregates compute across all available device tiles.

    Raises:
        ValueError: If `hint` is not one of the recognized performance hints.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.set_performance_hint("LATENCY")
    """
    valid_hints = {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"}
    hint_upper = hint.upper()
    if hint_upper not in valid_hints:
        raise ValueError(f"Invalid performance hint '{hint}'. Expected one of: {valid_hints}")

    if _C is not None and hasattr(_C, "set_performance_hint"):
        try:
            _C.set_performance_hint(hint_upper)
        except Exception as e:
            logger.warning(f"Failed to set performance hint '{hint_upper}': {e}")


def enable_turbo(enable: bool = True) -> None:
    """Enable or disable Intel NPU Turbo execution mode for burst workloads.

    When enabled, the NPU hardware clock frequency is boosted to its peak performance
    threshold for maximum compute density.

    Args:
        enable (bool, optional): True to enable Turbo boost, False to disable. Defaults to True.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.enable_turbo(True)
    """
    if _C is not None and hasattr(_C, "enable_turbo"):
        try:
            _C.enable_turbo(enable)
        except Exception as e:
            logger.warning(f"Failed to set turbo mode to {enable}: {e}")


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
    try:
        import openvino as ov
        core = ov.Core()
        return core.get_property("NPU", "FULL_DEVICE_NAME")
    except Exception:
        return "Intel(R) AI Boost (NPU)"


def get_device_properties(device: int = 0) -> Dict[str, Any]:
    """Get device configuration and hardware capability metadata.

    Args:
        device (int, optional): Device index. Defaults to 0.

    Returns:
        Dict[str, Any]: Dictionary containing device name, optimal streams, and supported properties.
    """
    props = {
        "name": get_device_name(device),
        "device_id": device,
        "is_available": is_available(),
    }
    if is_available():
        try:
            import openvino as ov
            core = ov.Core()
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


def synchronize(device: Optional[int] = None) -> None:
    """Wait for all outstanding asynchronous operations on the NPU to complete.

    Equivalent to `torch.cuda.synchronize()` for Intel NPU acceleration.

    Args:
        device (Optional[int], optional): Device index to synchronize. Defaults to None.
    """
    # OpenVINO / Level Zero handles synchronous fences on each InferRequest::infer() call
    pass


import contextlib


@contextlib.contextmanager
def turbo(enable: bool = True):
    """Context manager to temporarily enable or disable Intel NPU Turbo boost mode.

    Args:
        enable (bool, optional): Whether to enable Turbo boost. Defaults to True.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> with npu.turbo():
        ...     output = model(input_tensor)
    """
    try:
        enable_turbo(enable)
        yield
    finally:
        enable_turbo(not enable)


@contextlib.contextmanager
def performance_mode(hint: str = "LATENCY"):
    """Context manager to temporarily set NPU performance mode ('LATENCY' or 'THROUGHPUT').

    Args:
        hint (str, optional): Target performance mode. Defaults to 'LATENCY'.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> with npu.performance_mode("THROUGHPUT"):
        ...     results = [model(batch) for batch in batches]
    """
    try:
        set_performance_hint(hint)
        yield
    finally:
        set_performance_hint("LATENCY")


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


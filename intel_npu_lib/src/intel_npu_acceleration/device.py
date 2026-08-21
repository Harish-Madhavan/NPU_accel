"""
Intel NPU Device Management, Discovery, and Runtime Properties Configuration.
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

_EAGER_DEVICE = "NPU"


def is_available() -> bool:
    """
    Check if an Intel NPU device is available and accessible in the system.
    Returns:
        bool: True if Intel NPU is detected and functional, False otherwise.
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
    """
    Set low-level device configuration properties on the active NPU device.
    """
    if _C is not None and hasattr(_C, "set_property"):
        try:
            _C.set_property(key, value)
        except Exception as e:
            logger.warning(f"Failed to set property '{key}'='{value}': {e}")


def set_performance_hint(hint: str) -> None:
    """
    Configure runtime performance hint ("LATENCY", "THROUGHPUT", or "CUMULATIVE_THROUGHPUT").
    """
    if _C is not None and hasattr(_C, "set_performance_hint"):
        try:
            _C.set_performance_hint(hint)
        except Exception as e:
            logger.warning(f"Failed to set performance hint '{hint}': {e}")


def enable_turbo(enable: bool = True) -> None:
    """
    Enable or disable Intel NPU Turbo execution mode for burst workloads.
    """
    if _C is not None and hasattr(_C, "enable_turbo"):
        try:
            _C.enable_turbo(enable)
        except Exception as e:
            logger.warning(f"Failed to set turbo mode to {enable}: {e}")


def enable_sda(enable: bool = True) -> None:
    """
    Enable or disable Intel Smart Dispatch Acceleration (SDA).
    """
    if _C is not None and hasattr(_C, "enable_sda"):
        try:
            _C.enable_sda(enable)
        except Exception as e:
            logger.warning(f"Failed to set SDA mode to {enable}: {e}")


def set_eager_device(device_name: str) -> None:
    """
    Set the execution device for eager single-op executions ('NPU' or 'CPU').
    """
    global _EAGER_DEVICE
    _EAGER_DEVICE = device_name.upper()
    if _C is not None and hasattr(_C, "set_eager_device"):
        try:
            _C.set_eager_device(_EAGER_DEVICE)
        except Exception as e:
            logger.debug(f"Failed to set C++ eager device: {e}")


def get_eager_device() -> str:
    """
    Get the currently configured execution device for eager operations.
    """
    return _EAGER_DEVICE

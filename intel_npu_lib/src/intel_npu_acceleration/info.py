"""
Intel NPU System Diagnostics and Environment Introspection.

This module gathers diagnostic metadata about the host operating system, PyTorch runtime,
OpenVINO engine, oneAPI Level Zero driver, physical NPU hardware capabilities, and cache usage.
"""

import sys
import platform
import os
from typing import Dict, Any
import torch

try:
    import openvino as ov
    _OPENVINO_AVAILABLE = True
    _OPENVINO_VERSION = ov.__version__
except ImportError:
    _OPENVINO_AVAILABLE = False
    _OPENVINO_VERSION = "N/A"

import intel_npu_acceleration as npu


def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system, hardware, runtime, and NPU cache diagnostics.

    Returns:
        Dict[str, Any]: A structured dictionary containing:
            - `os`: Host operating system name.
            - `os_release`: OS release/version string.
            - `architecture`: CPU architecture (e.g., '64bit').
            - `python_version`: Active Python runtime version.
            - `pytorch_version`: Installed PyTorch version.
            - `openvino_available`: Boolean indicating if OpenVINO is importable.
            - `openvino_version`: OpenVINO release string.
            - `available_devices`: List of all detected OpenVINO devices (e.g., ['CPU', 'NPU']).
            - `device_details`: Per-device hardware metadata (e.g., full device names, optimal stream counts).
            - `npu_available`: Boolean indicating if Intel NPU is physically accessible.
            - `cache_dir`: Absolute path to active disk model cache.
            - `cache_size_mb`: Disk cache size in megabytes.
            - `cache_file_count`: Total number of cached binary blob files.
            - `in_memory_graph_cache_count`: Number of cached in-memory compiled model instances.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> info = npu.get_system_info()
        >>> print(info["npu_available"])
    """
    info = {}

    # Environment & OS
    info["os"] = platform.system()
    info["os_release"] = platform.release()
    info["architecture"] = platform.architecture()[0]
    info["python_version"] = sys.version.split()[0]

    # PyTorch & OpenVINO
    info["pytorch_version"] = torch.__version__
    info["openvino_available"] = _OPENVINO_AVAILABLE
    info["openvino_version"] = _OPENVINO_VERSION

    # Devices & Hardware Metadata
    info["device_details"] = {}
    if _OPENVINO_AVAILABLE:
        try:
            core = ov.Core()
            info["available_devices"] = core.available_devices
            for dev in core.available_devices:
                dev_meta = {}
                try:
                    dev_meta["full_name"] = core.get_property(dev, "FULL_DEVICE_NAME")
                except Exception:
                    pass
                try:
                    dev_meta["optimal_requests"] = core.get_property(
                        dev, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
                    )
                except Exception:
                    pass
                info["device_details"][dev] = dev_meta
        except Exception as e:
            info["available_devices"] = [f"Error: {e}"]
    else:
        info["available_devices"] = []

    # Intel NPU Hardware Status
    info["npu_available"] = npu.is_available()

    # Disk Cache Info
    cache_dir = npu.get_cache_dir()
    info["cache_dir"] = cache_dir if cache_dir else "None"
    total_bytes, file_count = npu.get_cache_size()
    info["cache_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    info["cache_file_count"] = file_count

    # In-memory graph cache
    try:
        from intel_npu_acceleration.frontend import _GRAPH_CACHE
        info["in_memory_graph_cache_count"] = len(_GRAPH_CACHE)
    except Exception:
        info["in_memory_graph_cache_count"] = 0

    return info


def print_info() -> None:
    """Format and print system diagnostic metrics to standard output.

    Prints a diagnostic summary of the Intel NPU status, OpenVINO
    environment, active device properties, and cache utilization.

    Examples:
        >>> import intel_npu_acceleration as npu
        >>> npu.print_info()
    """
    info = get_system_info()

    status_symbol = (
        "[AVAILABLE]" if info["npu_available"] else "[NOT DETECTED] (CPU Fallback active)"
    )

    banner = "=" * 65
    print(banner)
    print("      INTEL NPU ACCELERATION LIBRARY - DIAGNOSTIC REPORT")
    print(banner)
    print(f" NPU Hardware Status       : {status_symbol}")
    print(
        f" Operating System          : {info['os']} {info['os_release']} ({info['architecture']})"
    )
    print(f" Python Version            : {info['python_version']}")
    print(f" PyTorch Version           : {info['pytorch_version']}")
    print(
        f" OpenVINO Runtime          : {info['openvino_version']} (Available: {info['openvino_available']})"
    )
    print(
        f" Available OV Devices      : {', '.join(info['available_devices']) if info['available_devices'] else 'None'}"
    )
    if info.get("device_details"):
        for dev, meta in info["device_details"].items():
            name = meta.get("full_name", dev)
            reqs = meta.get("optimal_requests", "auto")
            print(f"   * {dev}: {name} (Optimal Streams: {reqs})")
    print("-" * 65)
    print(f" Disk Cache Directory      : {info['cache_dir']}")
    print(
        f" Disk Cache Usage          : {info['cache_size_mb']} MB ({info['cache_file_count']} files)"
    )
    print(
        f" In-Memory Graph Cache     : {info['in_memory_graph_cache_count']} active model graphs"
    )
    print(banner)


if __name__ == "__main__":
    print_info()

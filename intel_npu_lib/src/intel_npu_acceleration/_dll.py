"""
Windows dynamic-link library (DLL) resolution helper for Intel NPU & OpenVINO.

Ensures OpenVINO runtime libraries, Level Zero loader, and PyTorch C++ binaries
are in the Windows DLL search path (Python 3.8+) prior to importing native C++ extensions.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_DLL_RESOLVED = False


def setup_windows_dll_directories() -> None:
    """Register required PyTorch and OpenVINO DLL directories in Windows DLL search path."""
    global _DLL_RESOLVED
    if _DLL_RESOLVED or sys.platform != "win32":
        return

    # 1. PyTorch lib directory (c10.dll, torch_cpu.dll, etc.)
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception as e:
        logger.debug(f"PyTorch DLL directory setup skipped: {e}")

    # 2. OpenVINO runtime directories (openvino.dll, tbb, nGraph, plugins)
    try:
        import openvino

        ov_dir = os.path.dirname(openvino.__file__)
        for sub in [ov_dir, os.path.join(ov_dir, "libs"), os.path.join(ov_dir, "lib")]:
            if os.path.isdir(sub):
                os.add_dll_directory(sub)
    except Exception as e:
        logger.debug(f"OpenVINO DLL directory setup skipped: {e}")

    _DLL_RESOLVED = True

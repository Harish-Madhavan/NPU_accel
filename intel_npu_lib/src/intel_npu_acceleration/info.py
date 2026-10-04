"""
Intel NPU System Diagnostics and Environment Introspection.

This module gathers diagnostic metadata about the host operating system, PyTorch runtime,
OpenVINO engine, oneAPI Level Zero driver, physical NPU hardware capabilities, and cache usage.
"""

__all__ = ["get_system_info", "print_info", "self_test"]

import platform
import sys
from typing import Any

import torch

try:
    import openvino as ov

    _OPENVINO_AVAILABLE = True
    _OPENVINO_VERSION = ov.__version__
except ImportError:
    _OPENVINO_AVAILABLE = False
    _OPENVINO_VERSION = "N/A"
    ov = None  # type: ignore


def get_system_info() -> dict[str, Any]:
    """Gather comprehensive system, hardware, runtime, and NPU cache diagnostics."""
    # Lazy imports to avoid circular dependencies
    import intel_npu_acceleration as npu

    info: dict[str, Any] = {}

    # Environment & OS (platform.architecture() is deprecated — avoid it)
    info["os"] = platform.system()
    info["os_release"] = platform.release()
    info["architecture"] = platform.machine() or ("64bit" if sys.maxsize > 2**32 else "32bit")
    info["python_version"] = sys.version.split()[0]

    # PyTorch & OpenVINO
    info["pytorch_version"] = torch.__version__
    info["openvino_available"] = _OPENVINO_AVAILABLE
    info["openvino_version"] = _OPENVINO_VERSION

    # Devices & Hardware Metadata (reuse shared cached Core instance)
    info["device_details"] = {}
    if _OPENVINO_AVAILABLE and ov is not None:
        try:
            from intel_npu_acceleration.device import _get_ov_core

            core = _get_ov_core()
            if core is None:
                raise RuntimeError("OpenVINO Core unavailable")
            info["available_devices"] = core.available_devices
            for dev in core.available_devices:
                dev_meta: dict[str, Any] = {}
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
    try:
        info["npu_available"] = npu.is_available()
    except Exception:
        info["npu_available"] = False

    # Disk Cache Info
    try:
        cache_dir = npu.get_cache_dir()
        info["cache_dir"] = cache_dir if cache_dir else "None"
        total_bytes, file_count = npu.get_cache_size()
        info["cache_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        info["cache_file_count"] = file_count
    except Exception:
        info["cache_dir"] = "None"
        info["cache_size_mb"] = 0
        info["cache_file_count"] = 0

    # In-memory graph cache
    try:
        from intel_npu_acceleration.frontend.compiler import _GRAPH_CACHE

        info["in_memory_graph_cache_count"] = len(_GRAPH_CACHE)
    except Exception:
        info["in_memory_graph_cache_count"] = 0

    # PyTorch Backend & Accelerator diagnostics
    info["torch_npu_registered"] = hasattr(torch, "npu") and bool(getattr(torch.npu, "is_available", lambda: False)())
    info["torch_backends_npu_registered"] = hasattr(torch.backends, "npu") and bool(getattr(torch.backends.npu, "is_available", lambda: False)())
    try:
        info["dynamo_npu_registered"] = "npu" in torch._dynamo.list_backends()
    except Exception:
        info["dynamo_npu_registered"] = False

    if hasattr(torch, "accelerator"):
        try:
            import torch.accelerator as acc
            info["accelerator_available"] = acc.is_available()
            info["accelerator_device"] = str(acc.current_accelerator())
        except Exception:
            info["accelerator_available"] = False
            info["accelerator_device"] = "N/A"
    else:
        info["accelerator_available"] = False
        info["accelerator_device"] = "N/A"

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
    print(
        f" PyTorch Backend (torch.npu): {'[ACTIVE]' if info.get('torch_npu_registered') else '[INACTIVE]'}"
    )
    print(
        f" TorchDynamo Backend ('npu'): {'[REGISTERED]' if info.get('dynamo_npu_registered') else '[NOT REGISTERED]'}"
    )
    if info.get("accelerator_available"):
        print(f" Universal Accelerator     : {info.get('accelerator_device')}")
    print("-" * 65)
    print(f" Disk Cache Directory      : {info['cache_dir']}")
    print(
        f" Disk Cache Usage          : {info['cache_size_mb']} MB ({info['cache_file_count']} files)"
    )
    print(
        f" In-Memory Graph Cache     : {info['in_memory_graph_cache_count']} active model graphs"
    )
    print(banner)


def self_test(verbose: bool = True) -> bool:
    """Run an automated self-test of the Intel NPU acceleration backend.

    Verifies:
      1. NPU hardware availability and driver status
      2. PyTorch device integration ('npu' device placement)
      3. TorchDynamo graph compilation (torch.compile(..., backend='npu'))
      4. Numerical parity between NPU execution and standard CPU reference

    Returns:
        bool: True if all verification checks passed, False otherwise.
    """
    import time

    import intel_npu_acceleration as npu

    def log(msg: str):
        if verbose:
            print(msg)

    banner = "=" * 65
    log(banner)
    log("       INTEL NPU ACCELERATION - HARDWARE SELF-TEST")
    log(banner)

    all_passed = True
    start_time = time.perf_counter()

    # Step 1: Hardware Availability
    hw_avail = npu.is_available()
    dev_name = npu.get_device_name() if hw_avail else "CPU Fallback"
    if hw_avail:
        log(f" [PASS] Hardware Probing: Detected '{dev_name}'")
    else:
        log(" [WARN] Hardware Probing: Intel NPU not detected. Running CPU fallback.")

    # Step 2: PyTorch Device Integration
    try:
        t = torch.randn(2, 4, device="npu")
        log(f" [PASS] PyTorch Device: torch.Tensor allocated on '{t.device}'")
    except Exception as e:
        log(f" [FAIL] PyTorch Device: Failed to create tensor on npu device ({e})")
        all_passed = False

    # Step 3: Graph Compilation & Numerical Parity
    try:
        class MiniModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)
                self.act = torch.nn.GELU()

            def forward(self, x):
                return self.act(self.fc(x))

        m = MiniModel().eval()
        x = torch.randn(2, 8)
        expected = m(x)

        compiled = npu.compile(m, x)
        actual = compiled(x)

        if torch.allclose(actual, expected, atol=1e-2, rtol=1e-2):
            log(" [PASS] Graph Compilation & Numerics: NPU matches PyTorch CPU reference")
        else:
            diff = (actual - expected).abs().max().item()
            log(f" [FAIL] Graph Compilation & Numerics: Discrepancy max diff = {diff}")
            all_passed = False
    except Exception as e:
        log(f" [FAIL] Graph Compilation: {e}")
        all_passed = False

    elapsed = time.perf_counter() - start_time
    log("-" * 65)
    if all_passed:
        log(f" [SUCCESS] All self-tests passed successfully in {elapsed:.2f}s!")
    else:
        log(f" [FAILURE] Some verification checks failed ({elapsed:.2f}s).")
    log(banner)

    return all_passed


if __name__ == "__main__":
    print_info()
